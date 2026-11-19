namespace SafeBand.Core.Models.Sensors;

public sealed record AudioSample
{
    public double Rms { get; init; }

    public double Dc { get; init; }

    public double AverageRms { get; init; }

    public double MaxRms { get; init; }

    public int Blocks { get; init; }
}
