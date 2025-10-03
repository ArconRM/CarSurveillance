namespace CarSurveillance.Server.Entities;

public class UploadFile
{
    public string FileName { get; init; }

    public string ContentType { get; init; }

    public Stream Content { get; init; }
}