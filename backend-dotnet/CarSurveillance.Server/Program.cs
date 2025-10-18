using CarSurveillance.Server;
using CarSurveillance.Server.HttpService;
using CarSurveillance.Server.HttpService.Interfaces;
using CarSurveillance.Server.Options;
using CarSurveillance.Server.Repository;
using CarSurveillance.Server.Repository.Interfaces;
using CarSurveillance.Server.Service;
using CarSurveillance.Server.Service.Interfaces;
using DotNetEnv;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.EntityFrameworkCore;

Env.Load();

var builder = WebApplication.CreateBuilder(args);

var inferenceServerAddress = builder.Configuration.GetSection("ModelInferenceOptions")["ServerAddress"];

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddControllers();
builder.Services.AddSwaggerGen();

builder.Services.AddDbContext<CarSurveillanceContext>(options =>
    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.Configure<DataOptions>(
    builder.Configuration.GetSection("DataOptions"));

builder.Services.Configure<DataProcessingOptions>(
    builder.Configuration.GetSection("DataProcessingOptions"));

builder.Services.Configure<WeatherApiOptions>(options =>
{
    options.AppUrl = Environment.GetEnvironmentVariable("WEATHER_APP_URL");
    options.AppKey = Environment.GetEnvironmentVariable("WEATHER_APP_KEY");
    options.Lat = double.Parse(Environment.GetEnvironmentVariable("WEATHER_LAT"));
    options.Lon = double.Parse(Environment.GetEnvironmentVariable("WEATHER_LON"));
});

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

builder.Services.AddHttpClient<IWeatherHttpService, WeatherHttpService>(client =>
{
    var apiUrl = Environment.GetEnvironmentVariable("WEATHER_APP_URL");

    client.BaseAddress = new Uri(apiUrl);
    client.Timeout = TimeSpan.FromMinutes(1);
});

builder.Services.AddScoped<IFilesRepository, FilesRepository>();
builder.Services.AddScoped<IFilesService, FilesService>();

builder.Services.AddScoped<ICarPassRepository, CarPassRepository>();
builder.Services.AddScoped<ICarPassService, CarPassService>();

builder.Services.AddScoped<IWeatherRecordRepository, WeatherRecordRepository>();
builder.Services.AddScoped<IWeatherRecordService, WeatherRecordService>();

builder.Services.AddScoped<IModelInferenceService, ModelInferenceService>();

builder.Services.AddHostedService<DataProcessingBackgroundService>();

var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var dbContext = scope.ServiceProvider.GetRequiredService<CarSurveillanceContext>();
    dbContext.Database.Migrate();
}

app.UseExceptionHandlingMiddleware();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.MapControllers();
app.Run();