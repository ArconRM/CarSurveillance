using CarSurveillance.Server.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace CarSurveillance.Server.Repository.EntityConfiguration;

public class WeatherRecordEntityConfiguration : IEntityTypeConfiguration<WeatherRecord>
{
    public void Configure(EntityTypeBuilder<WeatherRecord> builder)
    {
        builder.HasKey(wr => wr.Uuid);

        builder.Property(wr => wr.Uuid)
            .ValueGeneratedOnAdd();
    }
}