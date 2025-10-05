using CarSurveillance.Server;
using CarSurveillance.Server.HttpService;
using CarSurveillance.Server.HttpService.Interfaces;
using CarSurveillance.Server.Options;
using CarSurveillance.Server.Repository;
using CarSurveillance.Server.Repository.Interfaces;
using CarSurveillance.Server.Service;
using CarSurveillance.Server.Service.Interfaces;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

var inferenceServerAddress = builder.Configuration.GetSection("ModelInferenceOptions")["ServerAddress"];

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddControllers();
builder.Services.AddSwaggerGen();

builder.Services.Configure<DataOptions>(
    builder.Configuration.GetSection("DataOptions"));

builder.Services.Configure<DataProcessingOptions>(
    builder.Configuration.GetSection("DataProcessingOptions"));

builder.WebHost.ConfigureKestrel(serverOptions => { serverOptions.Limits.MaxRequestBodySize = 1048576000; });

builder.Services.Configure<IISServerOptions>(options => { options.MaxRequestBodySize = 1048576000; });

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 1048576000;
    options.ValueLengthLimit = int.MaxValue;
    options.MemoryBufferThreshold = int.MaxValue;
});

builder.Services.AddHttpClient<IModelInferenceHttpService, ModelInferenceHttpService>(client =>
{
    client.BaseAddress = new Uri(inferenceServerAddress);
    client.Timeout = TimeSpan.FromHours(10);
});

builder.Services.AddScoped<IFilesRepository, FilesRepository>();
builder.Services.AddScoped<IFilesService, FilesService>();

builder.Services.AddHostedService<DataProcessingBackgroundService>();

var app = builder.Build();

app.UseExceptionHandlingMiddleware();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.MapControllers();
app.Run();