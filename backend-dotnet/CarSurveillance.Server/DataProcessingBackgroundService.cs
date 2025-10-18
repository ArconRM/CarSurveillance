using System.Diagnostics;
using CarSurveillance.Server.Options;
using CarSurveillance.Server.Service.Interfaces;
using Microsoft.Extensions.Options;

namespace CarSurveillance.Server;

public class DataProcessingBackgroundService : BackgroundService
{
    private readonly string _dataCropsPath;
    private readonly DataProcessingOptions _dataProcessingOptions;
    private readonly string _dataRawPath;
    private readonly string _dataResultsPath;
    private readonly ILogger<DataProcessingBackgroundService> _logger;

    private readonly IServiceScopeFactory _serviceScopeFactory;

    public DataProcessingBackgroundService(
        IOptions<DataProcessingOptions> dataProcessingOptions,
        IOptions<DataOptions> dataOptions,
        ILogger<DataProcessingBackgroundService> logger, IServiceScopeFactory serviceScopeFactory)
    {
        _dataProcessingOptions = dataProcessingOptions.Value;
        _dataRawPath = Path.Combine(dataOptions.Value.DataPath, dataOptions.Value.DataRawRelativePath);
        _dataCropsPath = Path.Combine(dataOptions.Value.DataPath, dataOptions.Value.DataCropsRelativePath);
        _dataResultsPath = Path.Combine(dataOptions.Value.DataPath, dataOptions.Value.DataResultRelativePath);
        _logger = logger;
        _serviceScopeFactory = serviceScopeFactory;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var hasAlreadyProcessed = false;

        using var scope = _serviceScopeFactory.CreateScope();
        var provider = scope.ServiceProvider;
        var weatherRecordService = provider.GetRequiredService<IWeatherRecordService>();
        var filesService = provider.GetRequiredService<IFilesService>();

        while (!stoppingToken.IsCancellationRequested)
        {
            if (IsProcessingHours() && !hasAlreadyProcessed)
            {
                await ProcessImagesAsync(stoppingToken);
                filesService.CleanRawDataFolder();
                hasAlreadyProcessed = true;
            }
            else
            {
                await weatherRecordService.SaveCurrentWeatherRecordFromApiAsync(stoppingToken);
                hasAlreadyProcessed = false;
            }

            await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
        }
    }

    private async Task ProcessImagesAsync(CancellationToken token)
    {
        using var scope = _serviceScopeFactory.CreateScope();
        var provider = scope.ServiceProvider;
        var modelInferenceService = provider.GetRequiredService<IModelInferenceService>();

        var stopwatch = Stopwatch.StartNew();
        _logger.LogInformation("Started processing images at {StartTime}", DateTime.Now);

        await modelInferenceService.CropToLicensePlatesAsync(
            _dataRawPath,
            _dataCropsPath,
            token);

        await modelInferenceService.RecognizeLicensePlatesAndSaveResults(
            _dataCropsPath,
            _dataResultsPath,
            token);

        stopwatch.Stop();
        _logger.LogInformation(
            "Finished processing images at {EndTime}. Duration: {Elapsed}",
            DateTime.Now,
            stopwatch.Elapsed
        );
    }

    private bool IsProcessingHours()
    {
        var now = GetLocalTime();
        return now.Hour >= _dataProcessingOptions.UploadingHourEnd ||
               now.Hour <= _dataProcessingOptions.UploadingHourStart;
    }

    private DateTime GetLocalTime()
    {
        var tz = TimeZoneInfo.FindSystemTimeZoneById(_dataProcessingOptions.TimeZone);
        return TimeZoneInfo.ConvertTime(DateTime.UtcNow, tz);
    }
}