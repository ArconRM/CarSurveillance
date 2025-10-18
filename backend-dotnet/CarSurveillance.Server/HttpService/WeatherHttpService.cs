using System.Globalization;
using System.Text.Json;
using CarSurveillance.Server.Dto.Responses;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.HttpService.Interfaces;
using CarSurveillance.Server.Options;
using Microsoft.Extensions.Options;

namespace CarSurveillance.Server.HttpService;

public class WeatherHttpService : IWeatherHttpService
{
    private readonly HttpClient _httpClient;
    private readonly WeatherApiOptions _weatherApiOptions;

    public WeatherHttpService(HttpClient httpClient, IOptions<WeatherApiOptions> weatherApiOptions)
    {
        _httpClient = httpClient;
        _weatherApiOptions = weatherApiOptions.Value;
    }

    public async Task<WeatherRecord> GetCurrentWeatherAsync(CancellationToken token)
    {
        var url = string.Format(CultureInfo.InvariantCulture,
            "/v1/current.json?key={0}&q={1},{2}&aqi=no",
            _weatherApiOptions.AppKey,
            _weatherApiOptions.Lat,
            _weatherApiOptions.Lon);

        var response = await _httpClient.GetAsync(url, token);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync(token);
        var deserialized = DeserializeWeatherData(json);
        return deserialized;
    }

    private async Task EnsureSuccessStatusCodeAsync(HttpResponseMessage response, CancellationToken token)
    {
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync(token);
            throw new HttpRequestException(
                $"API call failed with status code {response.StatusCode}: {errorContent}",
                null,
                response.StatusCode
            );
        }
    }

    private WeatherRecord DeserializeWeatherData(string jsonResponse)
    {
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        var weatherData = JsonSerializer.Deserialize<WeatherApiResponse>(jsonResponse, options);

        return new WeatherRecord
        {
            DateTime = DateTime.UtcNow,
            LocationName = weatherData.Location.Name,
            Latitude = weatherData.Location.Lat,
            Longitude = weatherData.Location.Lon,
            LocalTimeEpoch = weatherData.Location.LocaltimeEpoch,
            TemperatureC = weatherData.Current.TempC,
            WeatherCondition = weatherData.Current.Condition.Text,
            WeatherCode = weatherData.Current.Condition.Code,
            WindKph = weatherData.Current.WindKph,
            WindGustKph = weatherData.Current.GustKph,
            VisibilityKm = weatherData.Current.VisKm,
            PrecipitationMm = weatherData.Current.PrecipMm,
            Humidity = weatherData.Current.Humidity,
            CloudCover = weatherData.Current.Cloud,
            PressureMb = weatherData.Current.PressureMb,
            IsDay = weatherData.Current.IsDay == 1
        };
    }
}