namespace SafeBand.Core.Models.Sensors;

public sealed record MotionSample
{
    public double HeadingDeg { get; init; }

    public double RollDeg { get; init; }

    public double PitchDeg { get; init; }

    public double AccelXMetersPerSecondSquared { get; init; }

    public double AccelYMetersPerSecondSquared { get; init; }

    public double AccelZMetersPerSecondSquared { get; init; }

    public double GyroXRadiansPerSecond { get; init; }

    public double GyroYRadiansPerSecond { get; init; }

    public double GyroZRadiansPerSecond { get; init; }
}
