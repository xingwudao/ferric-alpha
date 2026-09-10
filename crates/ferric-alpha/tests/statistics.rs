use approx::assert_abs_diff_eq;
use ferric_alpha::{
    FerricAlphaError, average_rank, biased_excess_kurtosis, biased_skew, normal_qq,
    one_sample_t_test, pearson, sample_std, simple_ols, spearman, std_error,
};

const EPSILON: f64 = 1e-12;

fn assert_invalid_input(result: ferric_alpha::Result<impl core::fmt::Debug>) {
    assert!(matches!(result, Err(FerricAlphaError::InvalidInput { .. })));
}

#[test]
fn average_rank_is_one_based_and_averages_ties() {
    let ranks = average_rank(&[3.0, 1.0, 2.0, 2.0, 5.0]).unwrap();

    assert_eq!(ranks.len(), 5);
    assert_abs_diff_eq!(ranks[0], 4.0, epsilon = EPSILON);
    assert_abs_diff_eq!(ranks[1], 1.0, epsilon = EPSILON);
    assert_abs_diff_eq!(ranks[2], 2.5, epsilon = EPSILON);
    assert_abs_diff_eq!(ranks[3], 2.5, epsilon = EPSILON);
    assert_abs_diff_eq!(ranks[4], 5.0, epsilon = EPSILON);
}

#[test]
fn average_rank_accepts_empty_input() {
    let ranks = average_rank(&[]).unwrap();

    assert!(ranks.is_empty());
}

#[test]
fn statistics_reject_non_finite_input() {
    assert_invalid_input(average_rank(&[1.0, f64::NAN]));
    assert_invalid_input(pearson(&[1.0, f64::INFINITY], &[1.0, 2.0]));
    assert_invalid_input(spearman(&[1.0, 2.0], &[1.0, f64::NEG_INFINITY]));
    assert_invalid_input(simple_ols(&[1.0, 2.0], &[1.0, f64::NAN]));
    assert_invalid_input(sample_std(&[1.0, f64::INFINITY]));
    assert_invalid_input(std_error(&[1.0, f64::NEG_INFINITY]));
    assert_invalid_input(one_sample_t_test(&[1.0, f64::NAN], 0.0));
    assert_invalid_input(one_sample_t_test(&[1.0, 2.0], f64::INFINITY));
    assert_invalid_input(biased_skew(&[1.0, f64::NEG_INFINITY]));
    assert_invalid_input(biased_excess_kurtosis(&[f64::INFINITY]));
    assert_invalid_input(normal_qq(&[1.0, f64::NAN]));
}

#[test]
fn paired_statistics_reject_unequal_lengths() {
    assert_invalid_input(pearson(&[1.0, 2.0], &[1.0]));
    assert_invalid_input(spearman(&[1.0], &[1.0, 2.0]));
    assert_invalid_input(simple_ols(&[1.0, 2.0], &[1.0]));
}

#[test]
fn pearson_matches_hand_computed_fixture() {
    let actual = pearson(&[1.0, 2.0, 3.0], &[1.0, 5.0, 7.0]).unwrap();

    assert_abs_diff_eq!(actual.unwrap(), 0.9819805060619657, epsilon = EPSILON);
}

#[test]
fn pearson_returns_none_for_small_or_constant_input() {
    assert_eq!(pearson(&[], &[]).unwrap(), None);
    assert_eq!(pearson(&[1.0], &[2.0]).unwrap(), None);
    assert_eq!(pearson(&[1.0, 1.0], &[2.0, 3.0]).unwrap(), None);
    assert_eq!(pearson(&[1.0, 2.0], &[3.0, 3.0]).unwrap(), None);
}

#[test]
fn spearman_handles_positive_negative_and_tied_ranks() {
    let positive = spearman(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]).unwrap();
    let negative = spearman(&[1.0, 2.0, 3.0], &[30.0, 20.0, 10.0]).unwrap();
    let tied = spearman(&[1.0, 2.0, 2.0, 4.0], &[4.0, 1.0, 1.0, 2.0]).unwrap();

    assert_abs_diff_eq!(positive.unwrap(), 1.0, epsilon = EPSILON);
    assert_abs_diff_eq!(negative.unwrap(), -1.0, epsilon = EPSILON);
    assert_abs_diff_eq!(tied.unwrap(), -1.0 / 3.0, epsilon = EPSILON);
}

#[test]
fn simple_ols_matches_y_equals_one_plus_two_x() {
    let fit = simple_ols(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0]).unwrap();

    assert_eq!(fit.count, 4);
    assert_abs_diff_eq!(fit.alpha.unwrap(), 1.0, epsilon = EPSILON);
    assert_abs_diff_eq!(fit.beta.unwrap(), 2.0, epsilon = EPSILON);
}

#[test]
fn simple_ols_returns_none_for_small_or_constant_x() {
    let empty = simple_ols(&[], &[]).unwrap();
    let one = simple_ols(&[1.0], &[2.0]).unwrap();
    let constant = simple_ols(&[1.0, 1.0], &[2.0, 3.0]).unwrap();

    assert_eq!(empty.count, 0);
    assert_eq!(empty.alpha, None);
    assert_eq!(empty.beta, None);
    assert_eq!(one.count, 1);
    assert_eq!(one.alpha, None);
    assert_eq!(one.beta, None);
    assert_eq!(constant.count, 2);
    assert_eq!(constant.alpha, None);
    assert_eq!(constant.beta, None);
}

#[test]
fn sample_std_uses_n_minus_one() {
    let actual = sample_std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();

    assert_abs_diff_eq!(actual.unwrap(), 2.138089935299395, epsilon = EPSILON);
}

#[test]
fn sample_std_and_error_return_none_below_two_values() {
    assert_eq!(sample_std(&[]).unwrap(), None);
    assert_eq!(sample_std(&[1.0]).unwrap(), None);
    assert_eq!(std_error(&[]).unwrap(), None);
    assert_eq!(std_error(&[1.0]).unwrap(), None);

    let actual = std_error(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    assert_abs_diff_eq!(actual.unwrap(), 0.7559289460184544, epsilon = EPSILON);
}

#[test]
fn one_sample_t_test_matches_hand_computed_fixture() {
    let result = one_sample_t_test(&[1.0, 2.0, 3.0], 0.0).unwrap();

    assert_eq!(result.count, 3);
    assert_abs_diff_eq!(
        result.statistic.unwrap(),
        3.4641016151377544,
        epsilon = EPSILON
    );
    assert_abs_diff_eq!(
        result.p_value.unwrap(),
        0.07417990022744853,
        epsilon = EPSILON
    );
}

#[test]
fn one_sample_t_test_returns_null_statistics_when_undefined() {
    let empty = one_sample_t_test(&[], 0.0).unwrap();
    let one = one_sample_t_test(&[1.0], 0.0).unwrap();
    let constant = one_sample_t_test(&[2.0, 2.0, 2.0], 0.0).unwrap();

    assert_eq!(empty.count, 0);
    assert_eq!(empty.statistic, None);
    assert_eq!(empty.p_value, None);
    assert_eq!(one.count, 1);
    assert_eq!(one.statistic, None);
    assert_eq!(one.p_value, None);
    assert_eq!(constant.count, 3);
    assert_eq!(constant.statistic, None);
    assert_eq!(constant.p_value, None);
}

#[test]
fn biased_moments_match_population_central_moments() {
    let values = [1.0, 2.0, 4.0, 8.0];

    assert_abs_diff_eq!(
        biased_skew(&values).unwrap().unwrap(),
        0.6568077344996993,
        epsilon = EPSILON
    );
    assert_abs_diff_eq!(
        biased_excess_kurtosis(&values).unwrap().unwrap(),
        -1.0989792060491494,
        epsilon = EPSILON
    );
}

#[test]
fn biased_moments_return_none_for_empty_or_constant_input() {
    assert_eq!(biased_skew(&[]).unwrap(), None);
    assert_eq!(biased_skew(&[2.0, 2.0]).unwrap(), None);
    assert_eq!(biased_excess_kurtosis(&[]).unwrap(), None);
    assert_eq!(biased_excess_kurtosis(&[2.0, 2.0]).unwrap(), None);
}

#[test]
fn normal_qq_sorts_and_standardizes_values() {
    let points = normal_qq(&[3.0, 1.0, 2.0]).unwrap();

    assert_eq!(points.len(), 3);
    assert_abs_diff_eq!(points[0].probability, 1.0 / 6.0, epsilon = EPSILON);
    assert_abs_diff_eq!(points[0].theoretical, -0.967421566101701, epsilon = EPSILON);
    assert_abs_diff_eq!(points[0].observed, -1.0, epsilon = EPSILON);
    assert_abs_diff_eq!(points[1].probability, 0.5, epsilon = EPSILON);
    assert_abs_diff_eq!(points[1].theoretical, 0.0, epsilon = EPSILON);
    assert_abs_diff_eq!(points[1].observed, 0.0, epsilon = EPSILON);
    assert_abs_diff_eq!(points[2].probability, 5.0 / 6.0, epsilon = EPSILON);
    assert_abs_diff_eq!(points[2].theoretical, 0.967421566101701, epsilon = EPSILON);
    assert_abs_diff_eq!(points[2].observed, 1.0, epsilon = EPSILON);
}

#[test]
fn normal_qq_returns_empty_for_single_or_constant_input() {
    assert!(normal_qq(&[1.0]).unwrap().is_empty());
    assert!(normal_qq(&[1.0, 1.0, 1.0]).unwrap().is_empty());
}
