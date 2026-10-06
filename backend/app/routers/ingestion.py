"""LandSight AI - Live Project Ingestion Router.

Provides secure, transactional creation of infrastructure project records live,
initializing standard RFCTLARR statutory lifecycle milestones and customizable baseline records
for compensation, legal disputes, stakeholders, and rehabilitation.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from geoalchemy2.elements import WKTElement
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.compensation import CompensationRecord
from backend.app.models.enums import (
    CompensationStatusEnum,
    DataSourceEnum,
    DisputeStatusEnum,
    RRSchemeStatusEnum,
    StageNameEnum,
    StageStatusEnum,
)
from backend.app.models.legal import LegalDispute
from backend.app.models.project import Project
from backend.app.models.rehabilitation import RehabilitationProgress
from backend.app.models.stage import Stage
from backend.app.models.stakeholder import Stakeholder

router = APIRouter()


class ProjectCreateRequest(BaseModel):
    """Pydantic request model with strict validation for live project creation."""

    project_code: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique alphanumeric project code (e.g. NHAI-DL-PKG01)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Official infrastructure project name",
    )
    state: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="State administrative jurisdiction",
    )
    district: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="District where land acquisition occurs",
    )
    land_area_hectares: float = Field(
        ...,
        gt=0.0,
        description="Total land required for acquisition in hectares (strictly positive)",
    )
    affected_families_count: int = Field(
        ...,
        ge=0,
        description="Number of project-affected families (PAF, non-negative)",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees (-90 to 90)",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees (-180 to 180)",
    )
    data_source: Optional[Literal["real", "synthetic"]] = Field(
        default="real",
        description="Data provenance tier: 'real' (default) or 'synthetic'",
    )
    project_type: Optional[str] = Field(
        default="Infrastructure",
        description="Infrastructure sector classification",
    )
    notification_date: Optional[date] = Field(
        default=None,
        description="Date of Section 11 preliminary notification (defaults to today)",
    )
    status: Optional[str] = Field(
        default="active",
        description="Project lifecycle status",
    )
    stakeholder_responsiveness_score: Optional[float] = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Stakeholder responsiveness index score (0 to 100, default 70.0)",
    )
    compensation_disbursed_pct: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Disbursed compensation percentage (0 to 100, default 0.0)",
    )
    has_active_legal_dispute: Optional[bool] = Field(
        default=False,
        description="Active court stay order or litigation injunction (default False)",
    )

    model_config = ConfigDict(extra="forbid")


@router.post("/projects", status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Creates a new live infrastructure land acquisition project with:
    
    1. Project master record with PostGIS Point geography.
    2. The 5 standard statutory lifecycle stages (notification, survey, compensation, possession, rehabilitation).
    3. Starter records in compensation_records, legal_disputes, stakeholders, and rehabilitation_progress.
    """
    # 1. Check for duplicate project_code
    existing = db.query(Project).filter(Project.project_code == payload.project_code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project with project_code '{payload.project_code}' already exists.",
        )

    # 2. Determine provenance and dates
    source_enum = DataSourceEnum.real if payload.data_source == "real" else DataSourceEnum.synthetic
    notif_date = payload.notification_date or datetime.now(timezone.utc).date()
    point_geom = WKTElement(f"POINT({payload.longitude} {payload.latitude})", srid=4326)

    # 3. Create Project master record
    new_project = Project(
        project_code=payload.project_code,
        name=payload.name,
        state=payload.state,
        district=payload.district,
        project_type=payload.project_type or "Infrastructure",
        land_area_hectares=payload.land_area_hectares,
        affected_families_count=payload.affected_families_count,
        notification_date=notif_date,
        status=payload.status or "active",
        location=point_geom,
        data_source=source_enum,
    )
    db.add(new_project)
    db.flush()  # Generates new_project.id (UUID)

    # 4. Create the 5 standard lifecycle stage records matching RFCTLARR statutory ordering
    stages_spec = [
        (StageNameEnum.notification, 1, 60, StageStatusEnum.completed, notif_date, notif_date),
        (StageNameEnum.survey, 2, 90, StageStatusEnum.not_started, None, None),
        (StageNameEnum.compensation, 3, 120, StageStatusEnum.not_started, None, None),
        (StageNameEnum.possession, 4, 90, StageStatusEnum.not_started, None, None),
        (StageNameEnum.rehabilitation, 5, 150, StageStatusEnum.not_started, None, None),
    ]

    for stage_name, stage_order, planned_days, stage_status, start_dt, comp_dt in stages_spec:
        stg = Stage(
            project_id=new_project.id,
            stage_name=stage_name,
            stage_order=stage_order,
            planned_duration_days=planned_days,
            actual_duration_days=planned_days if stage_status == StageStatusEnum.completed else None,
            status=stage_status,
            start_date=start_dt,
            actual_completion_date=comp_dt,
            delay_days=0,
            data_source=source_enum,
        )
        db.add(stg)

    # 5. Insert starter compensation record using compensation_disbursed_pct
    allocated_amt = Decimal(str(round(max(500000.0, payload.land_area_hectares * 1000000.0), 2)))
    disbursed_pct = payload.compensation_disbursed_pct if payload.compensation_disbursed_pct is not None else 0.0
    disbursed_amt = Decimal(str(round(float(allocated_amt) * (disbursed_pct / 100.0), 2)))
    disbursed_cnt = int(round(payload.affected_families_count * (disbursed_pct / 100.0)))

    if disbursed_pct >= 100.0:
        comp_status = CompensationStatusEnum.fully_disbursed
    elif disbursed_pct > 0.0:
        comp_status = CompensationStatusEnum.partially_disbursed
    else:
        comp_status = CompensationStatusEnum.pending

    comp_record = CompensationRecord(
        project_id=new_project.id,
        total_amount_allocated=allocated_amt,
        total_amount_disbursed=disbursed_amt,
        beneficiaries_count=payload.affected_families_count,
        disbursed_count=disbursed_cnt,
        valuation_method="RFCTLARR 2013 Statutory Formula",
        status=comp_status,
        data_source=source_enum,
    )
    db.add(comp_record)

    # 6. Insert starter legal dispute record using has_active_legal_dispute
    has_dispute = bool(payload.has_active_legal_dispute)
    dispute_record = LegalDispute(
        project_id=new_project.id,
        case_number=f"{payload.project_code}-LIT-01" if has_dispute else f"{payload.project_code}-CLEAN-01",
        court_forum="High Court / Appellate Tribunal" if has_dispute else "District Civil Court",
        dispute_type="Land Acquisition Stay Petition" if has_dispute else "Title Clearance / Verification",
        stay_order_active=has_dispute,
        status=DisputeStatusEnum.stay_granted if has_dispute else DisputeStatusEnum.pending,
        filed_date=notif_date,
        delay_impact_estimate_days=90 if has_dispute else 0,
        petitioner_name="Landowners Association" if has_dispute else "None (Clean Title)",
        respondent_name="Competent Authority",
        summary=(
            "Interim injunction / stay order issued on possession proceedings."
            if has_dispute
            else "No active court injunction or stay order against acquisition."
        ),
        data_source=source_enum,
    )
    db.add(dispute_record)

    # 7. Insert starter stakeholder record using stakeholder_responsiveness_score
    stk_score = (
        payload.stakeholder_responsiveness_score
        if payload.stakeholder_responsiveness_score is not None
        else 70.0
    )
    stakeholder_record = Stakeholder(
        project_id=new_project.id,
        name=f"{payload.district} Land Acquisition Authority",
        role="Competent Authority",
        responsiveness_score=stk_score,
        last_contact_date=notif_date,
        notes="Primary nodal officer assigned for statutory RFCTLARR procedures.",
        data_source=source_enum,
    )
    db.add(stakeholder_record)

    # 8. Insert starter rehabilitation record (0% completed)
    rehab_record = RehabilitationProgress(
        project_id=new_project.id,
        total_families_eligible=payload.affected_families_count,
        families_resettled=0,
        monetary_allowance_disbursed=Decimal("0.00"),
        alternative_land_allotted_count=0,
        housing_units_constructed=0,
        housing_units_allotted=0,
        rr_scheme_status=RRSchemeStatusEnum.draft,
        completion_percentage=0.0,
        data_source=source_enum,
    )
    db.add(rehab_record)

    # 9. Commit transaction and refresh master project
    db.commit()
    db.refresh(new_project)

    return {
        "id": str(new_project.id),
        "project_code": new_project.project_code,
        "name": new_project.name,
        "state": new_project.state,
        "district": new_project.district,
        "project_type": new_project.project_type,
        "land_area_hectares": float(new_project.land_area_hectares),
        "affected_families_count": int(new_project.affected_families_count),
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "notification_date": new_project.notification_date.isoformat() if new_project.notification_date else None,
        "status": new_project.status,
        "data_source": new_project.data_source.value if hasattr(new_project.data_source, "value") else str(new_project.data_source),
        "created_at": new_project.created_at.isoformat() if new_project.created_at else None,
        "stages": [
            {
                "id": str(s.id),
                "stage_name": s.stage_name.value if hasattr(s.stage_name, "value") else str(s.stage_name),
                "stage_order": s.stage_order,
                "planned_duration_days": s.planned_duration_days,
                "actual_duration_days": s.actual_duration_days,
                "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                "delay_days": s.delay_days,
            }
            for s in sorted(new_project.stages, key=lambda x: x.stage_order)
        ],
        "compensation": {
            "total_amount_allocated": float(comp_record.total_amount_allocated),
            "total_amount_disbursed": float(comp_record.total_amount_disbursed),
            "compensation_disbursed_pct": disbursed_pct,
            "status": comp_record.status.value,
        },
        "disputes": [
            {
                "case_number": dispute_record.case_number,
                "stay_order_active": dispute_record.stay_order_active,
                "status": dispute_record.status.value,
                "delay_impact_estimate_days": dispute_record.delay_impact_estimate_days,
            }
        ],
        "stakeholders": [
            {
                "name": stakeholder_record.name,
                "role": stakeholder_record.role,
                "responsiveness_score": stakeholder_record.responsiveness_score,
            }
        ],
        "rehabilitation": {
            "completion_percentage": rehab_record.completion_percentage,
            "rr_scheme_status": rehab_record.rr_scheme_status.value,
        },
    }
