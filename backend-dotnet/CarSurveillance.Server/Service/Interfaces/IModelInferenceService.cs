namespace CarSurveillance.Server.Service.Interfaces;

public interface IModelInferenceService
{
    Task CropToLicensePlatesAsync(string rawDataPath, string resultDataPath, CancellationToken token);

    Task RecognizeLicensePlatesAndSaveResults(string cropsDataPath, string resultDataPath, CancellationToken token);
}