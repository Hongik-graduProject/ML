#!/bin/bash
set -e

# =============================================================================
# infra/deploy.sh — 원커맨드 배포 스크립트
# 실행 순서: Terraform → Ansible setup → Ansible run_pipeline
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/terraform"
ANSIBLE_DIR="${SCRIPT_DIR}/ansible"
TF_OUTPUT_FILE="${ANSIBLE_DIR}/inventory/tf_output.json"
HOSTS_FILE="${ANSIBLE_DIR}/inventory/hosts.yml"

# 색상 출력 헬퍼
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# 전제 조건 확인
check_prerequisites() {
    log_info "전제 조건 확인 중..."
    local missing=0

    for cmd in terraform ansible-playbook gcloud python3 jq; do
        if ! command -v "${cmd}" &>/dev/null; then
            log_error "필수 명령어 없음: ${cmd}"
            missing=1
        fi
    done

    if [[ ! -f "${TERRAFORM_DIR}/main.tf" ]]; then
        log_error "Terraform 설정 파일을 찾을 수 없습니다: ${TERRAFORM_DIR}/main.tf"
        missing=1
    fi

    if [[ ! -f "${ANSIBLE_DIR}/ansible.cfg" ]]; then
        log_error "Ansible 설정 파일을 찾을 수 없습니다: ${ANSIBLE_DIR}/ansible.cfg"
        missing=1
    fi

    if [[ ${missing} -ne 0 ]]; then
        log_error "전제 조건 확인 실패. 위 항목들을 해결한 후 다시 실행하세요."
        exit 1
    fi

    log_success "전제 조건 확인 완료"
}

# Terraform output에서 VM IP 추출하여 hosts.yml에 반영
update_ansible_inventory() {
    local tf_output_file="$1"

    if [[ ! -f "${tf_output_file}" ]]; then
        log_warn "tf_output.json 파일이 없습니다. hosts.yml을 수동으로 설정하세요."
        return 0
    fi

    local vm_ip
    vm_ip=$(jq -r '.vm_external_ip.value // empty' "${tf_output_file}" 2>/dev/null || true)

    if [[ -z "${vm_ip}" ]]; then
        log_warn "Terraform output에서 vm_external_ip를 찾을 수 없습니다."
        log_warn "hosts.yml의 ansible_host를 수동으로 설정하세요."
        return 0
    fi

    log_info "VM 외부 IP 감지: ${vm_ip}"

    # hosts.yml의 ansible_host 값을 실제 IP로 교체
    if command -v python3 &>/dev/null; then
        python3 - <<PYEOF
import yaml, re

with open('${HOSTS_FILE}', 'r') as f:
    content = f.read()

# REPLACE_WITH_VM_IP 또는 이전 IP 주소를 실제 IP로 교체
content = re.sub(
    r'(ansible_host:\s*")[^"]*(")',
    r'\g<1>${vm_ip}\g<2>',
    content
)
# lookup 기반 표현식도 직접 IP로 교체
content = re.sub(
    r"ansible_host:.*",
    'ansible_host: "${vm_ip}"',
    content
)

with open('${HOSTS_FILE}', 'w') as f:
    f.write(content)
print("hosts.yml 업데이트 완료: ansible_host = ${vm_ip}")
PYEOF
    else
        # fallback: sed 사용
        sed -i.bak "s/REPLACE_WITH_VM_IP/${vm_ip}/g" "${HOSTS_FILE}"
        log_info "hosts.yml ansible_host → ${vm_ip} (sed 사용)"
    fi

    # GCP 추가 출력값 환경변수 설정 (Ansible extra-vars용)
    local gcs_bucket gcp_project cloud_sql_instance
    gcs_bucket=$(jq -r '.gcs_bucket_name.value // empty' "${tf_output_file}" 2>/dev/null || true)
    gcp_project=$(jq -r '.gcp_project_id.value // empty' "${tf_output_file}" 2>/dev/null || true)
    cloud_sql_instance=$(jq -r '.cloud_sql_connection_name.value // empty' "${tf_output_file}" 2>/dev/null || true)

    # 이후 ansible-playbook 호출에서 사용할 extra-vars 파일 생성
    cat > "${ANSIBLE_DIR}/inventory/tf_vars.yml" <<VARSEOF
# Auto-generated from Terraform output — do not edit
gcs_bucket: "${gcs_bucket:-YOUR_GCS_BUCKET}"
gcp_project_id: "${gcp_project:-YOUR_GCP_PROJECT_ID}"
cloud_sql_instance: "${cloud_sql_instance:-YOUR_PROJECT:REGION:INSTANCE}"
VARSEOF

    log_success "Ansible 인벤토리 업데이트 완료 (VM IP: ${vm_ip})"
}

# ─────────────────────────────────────────
# 1단계: Terraform — 인프라 프로비저닝
# ─────────────────────────────────────────
echo ""
echo "=================================================================="
echo "=== 1단계: Terraform — GCP 인프라 프로비저닝                   ==="
echo "=================================================================="
log_info "시작 시각: $(date '+%Y-%m-%d %H:%M:%S')"

check_prerequisites

cd "${TERRAFORM_DIR}"

log_info "Terraform 초기화 중..."
terraform init -input=false

log_info "Terraform 계획 생성 중..."
terraform plan -out=tfplan -input=false

log_info "Terraform 인프라 적용 중 (자동 승인)..."
terraform apply -auto-approve tfplan

log_info "Terraform 출력값을 Ansible 인벤토리로 내보내는 중..."
terraform output -json > "${TF_OUTPUT_FILE}"

log_success "1단계 완료: GCP 인프라 프로비저닝 성공"
log_info "완료 시각: $(date '+%Y-%m-%d %H:%M:%S')"

# Ansible 인벤토리에 VM IP 반영
update_ansible_inventory "${TF_OUTPUT_FILE}"

# ─────────────────────────────────────────
# 2단계: Ansible — 서버 구성 + 앱 배포
# ─────────────────────────────────────────
echo ""
echo "=================================================================="
echo "=== 2단계: Ansible — 서버 구성 및 앱 배포                      ==="
echo "=================================================================="
log_info "시작 시각: $(date '+%Y-%m-%d %H:%M:%S')"

cd "${ANSIBLE_DIR}"

# Galaxy 컬렉션 설치 (없을 경우)
if [[ -f requirements.yml ]]; then
    log_info "Ansible Galaxy 컬렉션 설치 중..."
    ansible-galaxy collection install -r requirements.yml --ignore-errors
fi

EXTRA_VARS_ARGS=""
if [[ -f "${ANSIBLE_DIR}/inventory/tf_vars.yml" ]]; then
    EXTRA_VARS_ARGS="--extra-vars @${ANSIBLE_DIR}/inventory/tf_vars.yml"
fi

log_info "Ansible setup.yml 플레이북 실행 중..."
ansible-playbook playbooks/setup.yml \
    -i inventory/hosts.yml \
    ${EXTRA_VARS_ARGS} \
    --diff

log_success "2단계 완료: 서버 구성 및 앱 배포 성공"
log_info "완료 시각: $(date '+%Y-%m-%d %H:%M:%S')"

# ─────────────────────────────────────────
# 3단계: Ansible — ML 파이프라인 실행
# ─────────────────────────────────────────
echo ""
echo "=================================================================="
echo "=== 3단계: Ansible — ML 파이프라인 원격 실행                   ==="
echo "=================================================================="
log_info "시작 시각: $(date '+%Y-%m-%d %H:%M:%S')"

log_info "Ansible run_pipeline.yml 플레이북 실행 중..."
ansible-playbook playbooks/run_pipeline.yml \
    -i inventory/hosts.yml \
    ${EXTRA_VARS_ARGS}

log_success "3단계 완료: ML 파이프라인 실행 성공"
log_info "완료 시각: $(date '+%Y-%m-%d %H:%M:%S')"

# ─────────────────────────────────────────
# 최종 완료
# ─────────────────────────────────────────
echo ""
echo "=================================================================="
echo -e "${GREEN}=== 배포 완료 ===${NC}"
echo "=================================================================="
log_success "전체 배포 파이프라인이 성공적으로 완료되었습니다."
log_info "결과물은 GCS 버킷에서 확인하세요."
echo ""
