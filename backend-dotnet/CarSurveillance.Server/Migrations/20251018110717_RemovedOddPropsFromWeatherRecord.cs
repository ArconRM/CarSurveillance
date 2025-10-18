using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CarSurveillance.Server.Migrations
{
    /// <inheritdoc />
    public partial class RemovedOddPropsFromWeatherRecord : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "WindDirection",
                table: "WeatherRecords");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "WindDirection",
                table: "WeatherRecords",
                type: "text",
                nullable: false,
                defaultValue: "");
        }
    }
}
