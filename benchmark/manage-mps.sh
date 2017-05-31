#!/bin/bash

: ${MPSBASE:="/tmp/nvidia-mps-$USER"}
: ${MPSCONTROL:=nvidia-cuda-mps-control}

lookup()
{
  local MPS=$(pgrep -u "$UID" -fx "$MPSCONTROL -d" | head -1)
  if [ -z "$MPS" ]; then
    return 1
  fi
  local MPSPIPEDIR=$(tr '\0' '\n' </proc/$MPS/environ |
    sed -n 's/^CUDA_MPS_PIPE_DIRECTORY=\(.*\)/\1/p')
  if [ -n "$MPSPIPEDIR" ]; then
    dirname "$MPSPIPEDIR"
  else
    echo /tmp/nvidia-mps
  fi
}

start_()
{
  if lookup >/dev/null; then
    return
  fi

  CUDA_DEVICES=$@

  if [ -z "$CUDA_DEVICES" ]; then
    N_TESLA_DEVICES=$(nvidia-smi -L | grep Tesla | wc -l)
    CUDA_DEVICES="$(seq 0 $[$N_TESLA_DEVICES - 1])"
  fi

  if [ -z "$CUDA_DEVICES" ]; then
    return 1
  fi

  MPSDIRS="$(mktemp -d "$MPSBASE-XXXXXX")"

  declare -i j=0

  for i in $CUDA_DEVICES; do
    D="$MPSDIRS/$j"
    j=$[$j + 1]
    mkdir "$D"
    CUDA_VISIBLE_DEVICES=$i \
      CUDA_MPS_PIPE_DIRECTORY="$D" \
      CUDA_MPS_LOG_DIRECTORY="$D" \
      "$MPSCONTROL" -d >/dev/null
  done

  echo "$MPSDIRS"
}

start()
{
  MPSDIRS="$(flock -o /tmp -c "$0 start_ $@")" || return $?
  if [ -n "$MPSDIRS" ]; then
    echo MPSDIRS="$MPSDIRS"
  fi
}

stop_()
{
  MPSDIRS="$(lookup)" || return $?
  for i in "$MPSDIRS"/*; do
    CUDA_MPS_PIPE_DIRECTORY="$i" "$MPSCONTROL" <<<quit
  done
  rm -rf "$MPSDIRS"
}

stop()
{
  flock -o /tmp -c "$0 stop_"
}

wrap_()
{
  local MPSDIRS="$(lookup)" || return $?
  N=$(ls "$MPSDIRS/" | wc -l)
  I=${MV2_COMM_WORLD_LOCAL_RANK-${OMPI_COMM_WORLD_LOCAL_RANK-0}}
  CUDA_MPS_PIPE_DIRECTORY="$MPSDIRS/$[$I % $N]" "$@"
}

wrap()
{
  start >/dev/null
  wrap_ "$@"
  if [ -n "$MPSDIRS" ]; then
    stop
  fi
}

command="$1"
shift

case "$command" in
  lookup | start | start_ | stop | stop_ | wrap | wrap_)
    "$command" "$@"
    ;;
  *)
    echo "usage: $0 {start [DEVICE...] | stop | wrap COMMAND}"
esac
