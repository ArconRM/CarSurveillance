using CarSurveillance.Server.Entities;

namespace CarSurveillance.Server.Repository.Interfaces;

public interface IFilesRepository
{
    Task UploadZipAsync(Stream inputZip, CancellationToken token);

    Task UploadBatchAsync(List<UploadFile> images, CancellationToken token);

    void CleanRawDataFolder();
}