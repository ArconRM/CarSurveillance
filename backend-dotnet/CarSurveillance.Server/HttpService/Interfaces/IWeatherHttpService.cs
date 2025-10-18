using CarSurveillance.Server.Entities;

namespace CarSurveillance.Server.HttpService.Interfaces;

public interface IWeatherHttpService
{
    Task<WeatherRecord> GetCurrentWeatherAsync(CancellationToken token);
}