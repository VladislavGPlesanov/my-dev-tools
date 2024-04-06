   echo "Looking for: [ $1 ]"
   ps aux | grep $1
   echo "Its PIDs:"
   ps aux | grep $1 | awk '{print $2}'
   suspects=$(ps aux | grep $1 | awk '{print $2}')

   if [[ $2 == *"kill"* ]] && [[ -n "$suspects" ]]; then
      #kill -9 $(ps aux | grep $1 | awk '{print $2}')
      kill -9 $suspects
      echo "murder in process..."
      sleep 2
      survived=$(ps aux | grep $1 | awk '{print $2}')
      if [[ -z $survived ]]; then
          echo "NO survivors:"
      else
          echo "have survivors: ${survived}"
      fi
   else
      echo "nothing to kill..."
   fi

