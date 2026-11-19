using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.AI;

public sealed record ActivityResult : AiResult
{
    public ActivityState Activity { get; init; } = ActivityState.Unknown;
}
