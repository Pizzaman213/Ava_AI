#!/bin/bash
################################################################################
# Training Management Script
################################################################################
# Provides options to:
# - Optimize config with LR finder (automatically updates settings)
# - Start fresh training
# - Resume from checkpoint
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║               Ava Training Management System                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Select config file
echo -e "${CYAN}📋 Available Configs:${NC}"
echo "   1) small.yaml         (127M params, OPTIMIZED)"
echo "   2) tiny.yaml          (50M params)"
echo "   3) large.yaml         (350M params)"
echo "   4) small_corrected.yaml (older version)"
echo ""
read -p "Select config [1-4, default=1]: " config_choice
config_choice=${config_choice:-1}

case "$config_choice" in
    1) CONFIG="small.yaml" ;;
    2) CONFIG="tiny.yaml" ;;
    3) CONFIG="large.yaml" ;;
    4) CONFIG="small_corrected.yaml" ;;
    *) echo -e "${RED}Invalid choice. Using small.yaml${NC}"; CONFIG="small.yaml" ;;
esac

CONFIG_PATH="/project/code/configs/gpu/${CONFIG}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}❌ Config not found: ${CONFIG_PATH}${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Using config: ${CONFIG}"
echo

# Main menu
echo -e "${CYAN}🎯 Choose Action:${NC}"
echo "   1) Auto-optimize config + Start training (RECOMMENDED)"
echo "   2) Start fresh training (use current config)"
echo "   3) Resume from checkpoint"
echo "   4) Only run LR finder optimization (don't train)"
echo ""
read -p "Enter choice [1-4]: " choice

case "$choice" in
    1)
        echo
        echo -e "${YELLOW}🔍 Step 1: Running LR finder to optimize config...${NC}"
        echo
        /project/OPTIMIZE_CONFIG.sh "$CONFIG"

        if [ $? -eq 0 ]; then
            echo
            echo -e "${YELLOW}🚀 Step 2: Starting training with optimized config...${NC}"
            sleep 2
            cd /project/code
            python scripts/5_training/train.py --config "$CONFIG_PATH"
        else
            echo -e "${RED}❌ Optimization failed. Aborting training.${NC}"
            exit 1
        fi
        ;;

    2)
        echo
        echo -e "${YELLOW}🆕 Starting fresh training with current config...${NC}"
        echo -e "${BLUE}ℹ️  Note: Run option 4 first to optimize learning rate${NC}"
        sleep 2
        cd /project/code
        python scripts/5_training/train.py --config "$CONFIG_PATH"
        ;;

    3)
        echo
        echo -e "${CYAN}📂 Finding available checkpoints...${NC}"

        # Find latest checkpoint
        LATEST_CHECKPOINT=$(find /project/code/outputs -name "latest_model.pt" -o -name "model.pt" | sort -r | head -1)

        if [ -z "$LATEST_CHECKPOINT" ]; then
            echo -e "${RED}❌ No checkpoints found in outputs/runs/*/checkpoints/${NC}"
            echo
            echo "Available runs:"
            ls -1d /project/code/outputs/runs/run_* 2>/dev/null | xargs -n1 basename || echo "  None found"
            exit 1
        fi

        echo -e "${GREEN}✓${NC} Found checkpoint: ${LATEST_CHECKPOINT}"
        echo
        read -p "Use this checkpoint? [Y/n]: " use_checkpoint
        use_checkpoint=${use_checkpoint:-Y}

        if [[ "$use_checkpoint" =~ ^[Yy] ]]; then
            echo
            echo -e "${YELLOW}🔄 Resuming training from checkpoint...${NC}"
            sleep 2
            cd /project/code
            python scripts/5_training/train.py \
                --config "$CONFIG_PATH" \
                --resume-from-checkpoint "$LATEST_CHECKPOINT"
        else
            echo "Aborting."
            exit 0
        fi
        ;;

    4)
        echo
        echo -e "${YELLOW}🔍 Running LR finder optimization only...${NC}"
        echo
        /project/OPTIMIZE_CONFIG.sh "$CONFIG"

        if [ $? -eq 0 ]; then
            echo
            echo -e "${GREEN}✓ Optimization complete!${NC}"
            echo -e "${BLUE}ℹ️  Config has been updated. Run option 2 to start training.${NC}"
        else
            echo -e "${RED}❌ Optimization failed.${NC}"
            exit 1
        fi
        ;;

    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                          Done!                                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
