namespace CarSurveillance.Server.Service.Interfaces;

public interface IModelInferenceService
{
    Task CropToLicensePlatesAsync(string rawDataPath, string cropsDataPath, CancellationToken token);

    Task RecognizeLicensePlatesAndSaveResults(string cropsDataPath, string resultDataPath, CancellationToken token);
}