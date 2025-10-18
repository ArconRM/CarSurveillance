namespace CarSurveillance.Server.Dto.Requests;

public class DataCropInferenceRequest
{
    public string RawDataPath { get; set; }

    public string CropsDataPath { get; set; }
}