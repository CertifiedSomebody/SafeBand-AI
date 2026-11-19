using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.Sensors;

public sealed record EnvironmentSample
{
    public double TemperatureC { get; init; }

    public double HumidityPercent { get; init; }

    public double PressureHPa { get; init; }

    public double GasKOhm { get; init; }

    public SensorValidity Validity { get; init; } = SensorValidity.Unknown;
}
