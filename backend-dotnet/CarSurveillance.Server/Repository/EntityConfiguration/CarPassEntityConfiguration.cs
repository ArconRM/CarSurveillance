using CarSurveillance.Server.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace CarSurveillance.Server.Repository.EntityConfiguration;

public class CarPassEntityConfiguration : IEntityTypeConfiguration<CarPassRecord>
{
    public void Configure(EntityTypeBuilder<CarPassRecord> builder)
    {
        builder.HasKey(cp => cp.Uuid);

        builder.Property(cp => cp.Uuid)
            .ValueGeneratedOnAdd();
    }
}