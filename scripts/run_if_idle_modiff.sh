while true; do
    if ! ps aux | grep -q "[r]un_if_idle.sh"; then
        break
    fi
    date
    sleep 600
done

nohup zsh scripts/CG_CFGCG/run_experiment_sepguide_pro_shattered.sh & wait