#!/usr/bin/env bash
sync_basic(){
    remote_host=$1
    remote_dir=$2
    local_project_dir="/Users/mkakodka/Code/Research/RBM_V1/"
    remote_scp_path="$remote_host:$remote_dir"
    chmod +x "$local_project_dir""$d"
    for d in {'py','sh'}
    do
        echo "#################### Rsyncing $d to remote #################### "
        rsync -avz "$local_project_dir""$d" "$remote_scp_path"
    done

}

sync_basic "copa.cs.purdue.edu" "/scratch-data/mkakodka/rbm/"