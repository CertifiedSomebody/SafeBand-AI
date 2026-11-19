using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.AI;

public abstract record AiResult
{
    public AiResultStatus Status { get; init; } = AiResultStatus.Unknown;

    public double Confidence { get; init; }

    public DateTimeOffset? InferenceTimeUtc { get; init; }

    public string? ModelId { get; init; }

    public string? ModelVersion { get; init; }
}
