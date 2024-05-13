using ITensors: MPO, MPS, OpSum, expect, inner, siteinds
using ITensorTDVP: tdvp
using Observers: observer

function main()
  function heisenberg(N)
    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    return os
  end

  N = 10
  cutoff = 1e-12
  tau = 0.1
  ttotal = 1.0

  s = siteinds("S=1/2", N; conserve_qns=true)
  H = MPO(heisenberg(N), s)

  function step(; sweep, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return sweep
    end
    return nothing
  end

  function current_time(; current_time, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return current_time
    end
    return nothing
  end

  function measure_sz(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return expect(state, "Sz"; sites=N ÷ 2)
    end
    return nothing
  end

  function return_state(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return state
    end
    return nothing
  end

  obs = observer(
    "steps" => step, "times" => current_time, "states" => return_state, "Sz" => measure_sz
  )

  state = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  state_f = tdvp(
    H,
    -im * ttotal,
    state;
    time_step=-im * tau,
    cutoff,
    outputlevel=1,
    normalize=false,
    (observer!)=obs,
  )

  steps = obs.steps
  times = obs.times
  states = obs.states
  Sz = obs.Sz

  println("\nResults")
  println("=======")
  for n in 1:length(steps)
    print("step = ", steps[n])
    print(", time = ", round(times[n]; digits=3))
    print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(states[n], state)); digits=3))
    print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(states[n], state_f)); digits=3))
    print(", ⟨Sᶻ⟩ = ", round(Sz[n]; digits=3))
    println()
  end
  return nothing
end

main()
