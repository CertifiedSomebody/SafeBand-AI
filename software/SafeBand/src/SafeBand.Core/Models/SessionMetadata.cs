using SafeBand.Core.Enums;

namespace SafeBand.Core.Models;

public sealed record SessionMetadata
{
    public required string SessionId { get; init; }

    public string? SubjectId { get; init; }

    public ActivityState Activity { get; init; } = ActivityState.Unknown;

    public DataSourceType Source { get; init; } = DataSourceType.Unknown;

    public DateTimeOffset? StartTimeUtc { get; init; }

    public DateTimeOffset? EndTimeUtc { get; init; }

    public int ExpectedSampleIntervalMs { get; init; } = 200;

    public string? OriginalFileName { get; init; }
}
