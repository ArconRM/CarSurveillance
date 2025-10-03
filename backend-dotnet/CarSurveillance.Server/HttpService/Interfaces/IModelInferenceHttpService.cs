namespace CarSurveillance.Server.HttpService.Interfaces;

public interface IModelInferenceHttpService
{
    Task CropToLicensePlatesAsync(string rawDataPath, string resultDataPath, CancellationToken token);
}