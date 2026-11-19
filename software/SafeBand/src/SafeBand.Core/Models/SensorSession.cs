namespace SafeBand.Core.Models;

public sealed class SensorSession
{
    public required SessionMetadata Metadata { get; init; }

    public IList<SensorSample> Samples { get; } = new List<SensorSample>();
}
