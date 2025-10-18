using System.Reflection;
using CarSurveillance.Server.Entities;
using Microsoft.EntityFrameworkCore;

namespace CarSurveillance.Server.Repository;

public class CarSurveillanceContext : DbContext
{
    public CarSurveillanceContext(DbContextOptions options) : base(options) { }

    public DbSet<CarPassRecord> CarPassRecords { get; set; }

    public DbSet<WeatherRecord> WeatherRecords { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.ApplyConfigurationsFromAssembly(Assembly.GetExecutingAssembly());
    }
}