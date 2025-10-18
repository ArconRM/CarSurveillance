using CarSurveillance.Server.Core.Interfaces;

namespace CarSurveillance.Server.Entities;

public class CarPassRecord : IBaseEntity
{
    public DateTime DateTime { get; set; }

    public string Filename { get; set; }

    public string PlateTextRaw { get; set; }
    public double ConfidenceRaw { get; set; }

    public string PlateTextProcessed { get; set; }
    public double ConfidenceProcessed { get; set; }
    public Guid Uuid { get; set; }
}