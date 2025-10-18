using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CarSurveillance.Server.Migrations
{
    /// <inheritdoc />
    public partial class AddedDateTimePropToWeatherRecord : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<DateTime>(
                name: "DateTime",
                table: "WeatherRecords",
                type: "timestamp with time zone",
                nullable: false,
                defaultValue: new DateTime(1, 1, 1, 0, 0, 0, 0, DateTimeKind.Unspecified));
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "DateTime",
                table: "WeatherRecords");
        }
    }
}
