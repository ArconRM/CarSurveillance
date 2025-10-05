using System.Diagnostics;
using CarSurveillance.Server.HttpService.Interfaces;
using CarSurveillance.Server.Options;
using Microsoft.Extensions.Options;

namespace CarSurveillance.Server;

public class DataProcessingBackgroundService : BackgroundService
{
    private readonly string _dataCropsPath;
    private readonly DataProcessingOptions _dataProcessingOptions;
    private readonly string _dataRawPath;
    private readonly ILogger<DataProcessingBackgroundService> _logger;
    private readonly IModelInferenceHttpService _modelInferenceHttpService;

    public DataProcessingBackgroundService(
        IOptions<DataProcessingOptions> dataProcessingOptions,
        IOptions<DataOptions> dataOptions,
        IModelInferenceHttpService modelInferenceHttpService,
        ILogger<DataProcessingBackgroundService> logger)
    {
        _dataProcessingOptions = dataProcessingOptions.Value;
        _dataRawPath = Path.Combine(dataOptions.Value.DataPath, dataOptions.Value.DataRawRelativePath);
        _dataCropsPath = Path.Combine(dataOptions.Value.DataPath, dataOptions.Value.DataCropsRelativePath);
        _modelInferenceHttpService = modelInferenceHttpService;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            if (IsProcessingHours()) await ProcessImagesAsync(stoppingToken);

            await Task.Delay(TimeSpan.FromMinutes(30), stoppingToken);
        }
    }

    private async Task ProcessImagesAsync(CancellationToken token)
    {
        var stopwatch = Stopwatch.StartNew();
        _logger.LogInformation("Started processing images at {StartTime}", DateTime.Now);

        await _modelInferenceHttpService.CropToLicensePlatesAsync(
            _dataRawPath,
            _dataCropsPath,
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