namespace CarSurveillance.Server.Options;

public class WeatherApiOptions
{
    public string AppUrl { get; set; }

    public string AppKey { get; set; }

    public double Lat { get; set; }

    public double Lon { get; set; }
}