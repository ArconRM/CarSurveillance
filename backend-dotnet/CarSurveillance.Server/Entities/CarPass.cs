using CarSurveillance.Server.Core.Enums;
using CarSurveillance.Server.Core.Interfaces;

namespace CarSurveillance.Server.Entities;

public class CarPass : IBaseEntity
{
    public string NumberPlate { get; set; }

    public DateTime DateTime { get; set; }

    public WeatherCondition WeatherCondition { get; set; }
    public Guid Uuid { get; set; }
}