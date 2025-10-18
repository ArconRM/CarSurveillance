using CarSurveillance.Server.Core.BaseEntities;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.Repository.Interfaces;

namespace CarSurveillance.Server.Repository;

public class WeatherRecordRepository : BaseRepository<WeatherRecord>, IWeatherRecordRepository
{
    public WeatherRecordRepository(CarSurveillanceContext context) : base(context) { }
}