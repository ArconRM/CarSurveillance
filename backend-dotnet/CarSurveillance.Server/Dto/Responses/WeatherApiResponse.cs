using Newtonsoft.Json;

namespace CarSurveillance.Server.Dto.Responses;

public class WeatherApiResponse
{
    [JsonProperty("location")] public LocationData Location { get; set; }

    [JsonProperty("current")] public CurrentWeather Current { get; set; }
}

public class LocationData
{
    [JsonProperty("name")] public string Name { get; set; }

    [JsonProperty("lat")] public double Lat { get; set; }

    [JsonProperty("lon")] public double Lon { get; set; }

    [JsonProperty("localtime_epoch")] public long LocaltimeEpoch { get; set; }
}

public class CurrentWeather
{
    [JsonProperty("temp_c")] public double TempC { get; set; }

    [JsonProperty("condition")] public WeatherCondition Condition { get; set; }

    [JsonProperty("wind_kph")] public double WindKph { get; set; }

    [JsonProperty("wind_dir")] public string WindDir { get; set; }

    [JsonProperty("gust_kph")] public double GustKph { get; set; }

    [JsonProperty("vis_km")] public double VisKm { get; set; }

    [JsonProperty("precip_mm")] public double PrecipMm { get; set; }

    [JsonProperty("humidity")] public int Humidity { get; set; }

    [JsonProperty("cloud")] public int Cloud { get; set; }

    [JsonProperty("pressure_mb")] public double PressureMb { get; set; }

    [JsonProperty("is_day")] public int IsDay { get; set; }
}

public class WeatherCondition
{
    [JsonProperty("text")] public string Text { get; set; }

    [JsonProperty("code")] public int Code { get; set; }
}