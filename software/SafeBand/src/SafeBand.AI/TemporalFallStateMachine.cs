using SafeBand.Core.Models;

namespace SafeBand.AI;

public sealed class TemporalFallStateMachine
{
    private readonly double _suspectedThreshold;
    private readonly double _confirmationThreshold;
    private readonly double _recoveryThreshold;
    private readonly int _confirmationSamples;
    private readonly int _recoverySamples;
    private readonly long _recoveryCooldownMilliseconds;
    private readonly long _maximumTimingGapMilliseconds;

    private long? _lastElapsedMilliseconds;
    private long? _recoveryStartedMilliseconds;
    private int _highProbabilitySamples;
    private int _lowProbabilitySamples;
    private bool _eventLatched;

    public TemporalFallStateMachine(
        double suspectedThreshold = 0.65,
        double confirmationThreshold = 0.8,
        double recoveryThreshold = 0.35,
        int confirmationSamples = 3,
        int recoverySamples = 5,
        long recoveryCooldownMilliseconds = 3000,
        long maximumTimingGapMilliseconds = 1000)
    {
        if (!double.IsFinite(suspectedThreshold)
            || !double.IsFinite(confirmationThreshold)
            || !double.IsFinite(recoveryThreshold)
            || suspectedThreshold is < 0 or > 1
            || confirmationThreshold is < 0 or > 1
            || recoveryThreshold is < 0 or > 1
            || recoveryThreshold >= suspectedThreshold
            || suspectedThreshold > confirmationThreshold)
        {
            throw new ArgumentOutOfRangeException(
                nameof(suspectedThreshold),
                "Fall probability thresholds must satisfy 0 <= recovery < suspected <= confirmation <= 1.");
        }

        if (confirmationSamples < 1
            || recoverySamples < 1
            || recoveryCooldownMilliseconds < 0
            || maximumTimingGapMilliseconds < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(confirmationSamples),
                "Sample counts and timing limits must be positive.");
        }

        _suspectedThreshold = suspectedThreshold;
        _confirmationThreshold = confirmationThreshold;
        _recoveryThreshold = recoveryThreshold;
        _confirmationSamples = confirmationSamples;
        _recoverySamples = recoverySamples;
        _recoveryCooldownMilliseconds = recoveryCooldownMilliseconds;
        _maximumTimingGapMilliseconds = maximumTimingGapMilliseconds;
    }

    public TemporalFallDecision Update(
        SensorSample sample,
        double? fallProbability,
        string? modelVersion = null)
    {
        ArgumentNullException.ThrowIfNull(sample);
        if (_lastElapsedMilliseconds is not null
            && sample.ElapsedMilliseconds <= _lastElapsedMilliseconds.Value)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sample),
                "Elapsed time must increase strictly for each state-machine update.");
        }
        _lastElapsedMilliseconds = sample.ElapsedMilliseconds;

        if (fallProbability is not null
            && (!double.IsFinite(fallProbability.Value)
                || fallProbability.Value is < 0 or > 1))
        {
            throw new ArgumentOutOfRangeException(
                nameof(fallProbability),
                "Fall probability must be finite and between 0 and 1.");
        }

        var quality = sample.Quality;
        var validMotionAndTiming =
            sample.Motion is not null
            && quality is { HasValidMotion: true, TimingWithinExpectedRange: true }
            && (quality.DeltaMilliseconds is null
                || quality.DeltaMilliseconds <= _maximumTimingGapMilliseconds)
            && MotionValuesAreFinite(sample);

        if (!validMotionAndTiming || fallProbability is null)
        {
            _highProbabilitySamples = 0;
            _lowProbabilitySamples = 0;
            if (_eventLatched)
            {
                return Decision(
                    _recoveryStartedMilliseconds is null
                        ? FallState.Confirmed
                        : FallState.Recovery,
                    fallProbability,
                    newlyConfirmed: false,
                    validMotionAndTiming,
                    modelVersion);
            }

            return Decision(
                FallState.InsufficientData,
                fallProbability,
                newlyConfirmed: false,
                validMotionAndTiming,
                modelVersion);
        }

        var probability = fallProbability.Value;
        if (_eventLatched)
        {
            return UpdateLatchedEvent(
                sample.ElapsedMilliseconds,
                probability,
                validMotionAndTiming,
                modelVersion);
        }

        if (probability >= _confirmationThreshold)
        {
            _highProbabilitySamples++;
            if (_highProbabilitySamples >= _confirmationSamples)
            {
                _eventLatched = true;
                _lowProbabilitySamples = 0;
                return Decision(
                    FallState.Confirmed,
                    probability,
                    newlyConfirmed: true,
                    validMotionAndTiming,
                    modelVersion);
            }
        }
        else
        {
            _highProbabilitySamples = 0;
        }

        var state = probability >= _suspectedThreshold
            || _highProbabilitySamples > 0
            ? FallState.Suspected
            : FallState.Monitoring;
        return Decision(
            state,
            probability,
            newlyConfirmed: false,
            validMotionAndTiming,
            modelVersion);
    }

    private TemporalFallDecision UpdateLatchedEvent(
        long elapsedMilliseconds,
        double probability,
        bool validMotionAndTiming,
        string? modelVersion)
    {
        if (_recoveryStartedMilliseconds is null)
        {
            if (probability <= _recoveryThreshold)
            {
                _lowProbabilitySamples++;
                if (_lowProbabilitySamples >= _recoverySamples)
                {
                    _recoveryStartedMilliseconds = elapsedMilliseconds;
                    _lowProbabilitySamples = 0;
                }
            }
            else
            {
                _lowProbabilitySamples = 0;
            }

            return Decision(
                FallState.Confirmed,
                probability,
                newlyConfirmed: false,
                validMotionAndTiming,
                modelVersion);
        }

        if (probability > _recoveryThreshold)
        {
            _recoveryStartedMilliseconds = null;
            _lowProbabilitySamples = 0;
            return Decision(
                FallState.Confirmed,
                probability,
                newlyConfirmed: false,
                validMotionAndTiming,
                modelVersion);
        }

        if (elapsedMilliseconds - _recoveryStartedMilliseconds.Value
            >= _recoveryCooldownMilliseconds)
        {
            _eventLatched = false;
            _recoveryStartedMilliseconds = null;
            _lowProbabilitySamples = 0;
            _highProbabilitySamples = 0;
            return Decision(
                FallState.Monitoring,
                probability,
                newlyConfirmed: false,
                validMotionAndTiming,
                modelVersion);
        }

        return Decision(
            FallState.Recovery,
            probability,
            newlyConfirmed: false,
            validMotionAndTiming,
            modelVersion);
    }

    private static bool MotionValuesAreFinite(SensorSample sample)
    {
        var motion = sample.Motion!;
        return double.IsFinite(motion.AccelXMetersPerSecondSquared)
            && double.IsFinite(motion.AccelYMetersPerSecondSquared)
            && double.IsFinite(motion.AccelZMetersPerSecondSquared)
            && double.IsFinite(motion.GyroXRadiansPerSecond)
            && double.IsFinite(motion.GyroYRadiansPerSecond)
            && double.IsFinite(motion.GyroZRadiansPerSecond)
            && double.IsFinite(motion.HeadingDeg)
            && double.IsFinite(motion.RollDeg)
            && double.IsFinite(motion.PitchDeg);
    }

    private static TemporalFallDecision Decision(
        FallState state,
        double? probability,
        bool newlyConfirmed,
        bool validMotionAndTiming,
        string? modelVersion) =>
        new()
        {
            State = state,
            FallProbability = probability,
            NewlyConfirmed = newlyConfirmed,
            HasValidMotionAndTiming = validMotionAndTiming,
            ModelVersion = modelVersion
        };
}
