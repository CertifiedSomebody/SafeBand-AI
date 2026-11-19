using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.AI;

public sealed record AudioResult : AiResult
{
    public string? EventLabel { get; init; }

    public double EventProbability { get; init; }
}
