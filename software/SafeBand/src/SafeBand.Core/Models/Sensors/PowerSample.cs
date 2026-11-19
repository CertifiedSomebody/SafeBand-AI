namespace SafeBand.Core.Models.Sensors;

public sealed record PowerSample
{
    public double BatteryVoltageV { get; init; }

    public double BatteryPercent { get; init; }

    public bool ChargerConnected { get; init; }
}
