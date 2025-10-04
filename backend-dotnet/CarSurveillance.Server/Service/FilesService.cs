using CarSurveillance.Server.Entities;
using CarSurveillance.Server.Repository.Interfaces;
using CarSurveillance.Server.Service.Interfaces;

namespace CarSurveillance.Server.Service;

public class FilesService : IFilesService
{
    private readonly IFilesRepository _filesRepository;

    public FilesService(IFilesRepository filesRepository)
    {
        _filesRepository = filesRepository;
    }

    public async Task UploadZipAsync(Stream inputZip, CancellationToken token)
    {
        await _filesRepository.UploadZipAsync(inputZip, token);
    }

    public async Task UploadBatchAsync(List<UploadFile> images, CancellationToken token)
    {
        await _filesRepository.UploadBatchAsync(images, token);
    }

    public void CleanRawDataFolder()
    {
        _filesRepository.CleanRawDataFolder();
    }
}