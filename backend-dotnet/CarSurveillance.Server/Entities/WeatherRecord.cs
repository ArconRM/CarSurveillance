using CarSurveillance.Server.Core.Interfaces;

namespace CarSurveillance.Server.Entities;

public class WeatherRecord : IBaseEntity
{
    public DateTime DateTime { get; set; }

    // Location data
    public string LocationName { get; set; }
    public double Latitude { get; set; }
    public double Longitude { get; set; }
    public long LocalTimeEpoch { get; set; }

    // Weather conditions that affect driving
    public double TemperatureC { get; set; }
    public string WeatherCondition { get; set; }
    public int WeatherCode { get; set; }

    // Wind conditions
    public double WindKph { get; set; }
    public double WindGustKph { get; set; }

    // Visibility and precipitation
    public double VisibilityKm { get; set; }
    public double PrecipitationMm { get; set; }
    public int Humidity { get; set; }
    public int CloudCover { get; set; }

    // Road conditions
    public double PressureMb { get; set; }
    public bool IsDay { get; set; }
    public Guid Uuid { get; set; }
}