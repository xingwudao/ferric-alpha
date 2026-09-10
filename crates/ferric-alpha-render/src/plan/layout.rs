use std::collections::BTreeMap;

use crate::{MAX_LOGICAL_HEIGHT, MAX_PANELS, RenderError, RenderResult};

use super::PanelPlan;

pub(crate) const OUTER_MARGIN: u32 = 24;
pub(crate) const TITLE_AREA: u32 = 56;
pub(crate) const ROW_GAP: u32 = 24;

pub(crate) const CHART_HEIGHT: u32 = 320;
pub(crate) const DISTRIBUTION_HEIGHT: u32 = 360;

pub(crate) fn assign_layout(
    width: u32,
    mut panels: Vec<PanelPlan>,
) -> RenderResult<Vec<PanelPlan>> {
    if panels.len() > MAX_PANELS {
        return Err(RenderError::LayoutOverflow);
    }

    let wide = width >= 960;
    let mut row = 0;
    let mut pending_half: Option<usize> = None;

    for index in 0..panels.len() {
        if !wide || panels[index].column_span == 12 {
            if pending_half.take().is_some() {
                row += 1;
            }
            panels[index].row = row;
            panels[index].column = 0;
            panels[index].column_span = 12;
            row += 1;
            continue;
        }

        if let Some(left_index) = pending_half.take() {
            if panels[left_index].height == panels[index].height {
                panels[index].row = panels[left_index].row;
                panels[index].column = 6;
            } else {
                row += 1;
                panels[index].row = row;
                panels[index].column = 0;
                pending_half = Some(index);
            }
        } else {
            panels[index].row = row;
            panels[index].column = 0;
            pending_half = Some(index);
        }

        if panels[index].column == 6 {
            row += 1;
        }
    }

    Ok(panels)
}

pub(crate) fn total_height(panels: &[PanelPlan]) -> RenderResult<u32> {
    let mut rows = BTreeMap::<u32, u32>::new();
    for panel in panels {
        rows.entry(panel.row).or_insert(panel.height);
    }
    let mut total = OUTER_MARGIN
        .checked_add(TITLE_AREA)
        .and_then(|value| value.checked_add(OUTER_MARGIN))
        .ok_or(RenderError::LayoutOverflow)?;
    for (index, height) in rows.values().enumerate() {
        if index > 0 {
            total = total
                .checked_add(ROW_GAP)
                .ok_or(RenderError::LayoutOverflow)?;
        }
        total = total
            .checked_add(*height)
            .ok_or(RenderError::LayoutOverflow)?;
    }
    if total > MAX_LOGICAL_HEIGHT {
        return Err(RenderError::LayoutOverflow);
    }
    Ok(total)
}
