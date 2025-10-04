using CarSurveillance.Server.Entities;

namespace CarSurveillance.Server.Service.Interfaces;

public interface IFilesService
{
    Task UploadZipAsync(Stream inputZip, CancellationToken token);

    Task UploadBatchAsync(List<UploadFile> images, CancellationToken token);

    void CleanRawDataFolder();
}