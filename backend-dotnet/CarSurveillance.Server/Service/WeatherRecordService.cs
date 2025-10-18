using CarSurveillance.Server.Core.BaseEntities;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.HttpService.Interfaces;
using CarSurveillance.Server.Repository.Interfaces;
using CarSurveillance.Server.Service.Interfaces;

namespace CarSurveillance.Server.Service;

public class WeatherRecordService : BaseService<WeatherRecord>, IWeatherRecordService
{
    private readonly IWeatherRecordRepository _repository;
    private readonly IWeatherHttpService _weatherHttpService;

    public WeatherRecordService(
        IWeatherRecordRepository repository,
        IWeatherHttpService weatherHttpService) : base(repository)
    {
        _repository = repository;
        _weatherHttpService = weatherHttpService;
    }

    public async Task<WeatherRecord> SaveCurrentWeatherRecordFromApiAsync(CancellationToken token)
    {
        var weatherRecord = await _weatherHttpService.GetCurrentWeatherAsync(token);
        return await _repository.CreateAsync(weatherRecord, token);
    }
}