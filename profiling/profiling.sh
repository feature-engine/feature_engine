mkdir -p profiles/code_profiles

file="$@"

pyinstrument -r html -o profiles/code_profiles/performance_profile_$(date "+%Y.%m.%d-%H:%M").html $file

pyinstrument -r speedscope -o profiles/code_profiles/speedscope_$(date "+%Y.%m.%d-%H:%M").json $file
