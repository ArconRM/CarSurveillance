using System.IO.Compression;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.Options;
using CarSurveillance.Server.Repository.Interfaces;
using Microsoft.Extensions.Options;

namespace CarSurveillance.Server.Repository;

public class FilesRepository : IFilesRepository
{
    private readonly string _dataPath;
    private readonly string _dataRawPath;
    private readonly ILogger<FilesRepository> _logger;

    public FilesRepository(IOptions<DataOptions> options, ILogger<FilesRepository> logger)
    {
        _dataPath = options.Value.DataPath;
        _dataRawPath = Path.Combine(_dataPath, "raw");
        _logger = logger;
    }

    public async Task UploadZipAsync(Stream inputZip, CancellationToken token)
    {
        CleanDataFolder();
        using var archive = new ZipArchive(inputZip, ZipArchiveMode.Read);

        foreach (var entry in archive.Entries)
        {
            if (string.IsNullOrEmpty(entry.Name))
                continue;

            var extension = Path.GetExtension(entry.Name).ToLowerInvariant();
            if (extension is not (".jpg" or ".jpeg" or ".png" or ".gif" or ".bmp" or ".webp"))
                continue;

            var destinationPath = Path.Combine(_dataRawPath, entry.Name);

            await using var entryStream = entry.Open();
            await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write);
            await entryStream.CopyToAsync(fileStream, token);
            _logger.LogInformation($"Saved {entry.Name}");
        }
    }

    public async Task UploadBatchAsync(List<UploadFile> images, CancellationToken token)
    {
        CleanDataFolder();

        foreach (var image in images)
        {
            var extension = Path.GetExtension(image.FileName).ToLowerInvariant();
            if (extension is not (".jpg" or ".jpeg" or ".png" or ".gif" or ".bmp" or ".webp"))
                continue;

            var destinationPath = Path.Combine(_dataRawPath, image.FileName);

            await using var stream = new FileStream(destinationPath, FileMode.Create);
            await image.Content.CopyToAsync(stream, token);
            _logger.LogInformation($"Saved {image.FileName}");
        }
    }

    private void CleanDataFolder()
    {
        if (Directory.Exists(_dataRawPath))
        {
            var files = Directory.GetFiles(_dataRawPath);
            var directories = Directory.GetDirectories(_dataRawPath);

            foreach (var file in files) File.Delete(file);

            foreach (var dir in directories) Directory.Delete(dir, true);
        }
        else
        {
            Directory.CreateDirectory(_dataRawPath);
        }
    }
}