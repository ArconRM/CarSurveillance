using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CarSurveillance.Server.Migrations
{
    /// <inheritdoc />
    public partial class Initial : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "CarPassRecords",
                columns: table => new
                {
                    Uuid = table.Column<Guid>(type: "uuid", nullable: false),
                    DateTime = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    Filename = table.Column<string>(type: "text", nullable: false),
                    PlateTextRaw = table.Column<string>(type: "text", nullable: false),
                    ConfidenceRaw = table.Column<double>(type: "double precision", nullable: false),
                    PlateTextProcessed = table.Column<string>(type: "text", nullable: false),
                    ConfidenceProcessed = table.Column<double>(type: "double precision", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_CarPassRecords", x => x.Uuid);
                });

            migrationBuilder.CreateTable(
                name: "WeatherRecords",
                columns: table => new
                {
                    Uuid = table.Column<Guid>(type: "uuid", nullable: false),
                    LocationName = table.Column<string>(type: "text", nullable: false),
                    Latitude = table.Column<double>(type: "double precision", nullable: false),
                    Longitude = table.Column<double>(type: "double precision", nullable: false),
                    LocalTimeEpoch = table.Column<long>(type: "bigint", nullable: false),
                    TemperatureC = table.Column<double>(type: "double precision", nullable: false),
                    WeatherCondition = table.Column<string>(type: "text", nullable: false),
                    WeatherCode = table.Column<int>(type: "integer", nullable: false),
                    WindKph = table.Column<double>(type: "double precision", nullable: false),
                    WindDirection = table.Column<string>(type: "text", nullable: false),
                    WindGustKph = table.Column<double>(type: "double precision", nullable: false),
                    VisibilityKm = table.Column<double>(type: "double precision", nullable: false),
                    PrecipitationMm = table.Column<double>(type: "double precision", nullable: false),
                    Humidity = table.Column<int>(type: "integer", nullable: false),
                    CloudCover = table.Column<int>(type: "integer", nullable: false),
                    PressureMb = table.Column<double>(type: "double precision", nullable: false),
                    IsDay = table.Column<bool>(type: "boolean", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_WeatherRecords", x => x.Uuid);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "CarPassRecords");

            migrationBuilder.DropTable(
                name: "WeatherRecords");
        }
    }
}
