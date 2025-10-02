FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY ["CarSurveillance.Server/CarSurveillance.Server.csproj", "CarSurveillance.Server/"]

RUN dotnet restore "CarSurveillance.Server/CarSurveillance.Server.csproj"

COPY ["CarSurveillance.Server/", "CarSurveillance.Server/"]

RUN dotnet publish "CarSurveillance.Server/CarSurveillance.Server.csproj" -c Release -o /app/publish --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final

WORKDIR /app
COPY --from=build /app/publish .
ENTRYPOINT ["dotnet", "CarSurveillance.Server.dll"]