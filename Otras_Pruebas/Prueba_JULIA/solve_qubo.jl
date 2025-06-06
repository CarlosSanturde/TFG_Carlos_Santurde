# Agregar la ruta de QuantumAnnealing.jl-main a LOAD_PATH
push!(LOAD_PATH, "C:/Users/carlo/TFG/Q-Seg/QuantumAnnealing.jl-main")

using QuantumAnnealing
using DelimitedFiles

# 1. Cargar la matriz QUBO desde qubo_matrix.csv
qubo_matrix = readdlm("qubo_matrix.csv", ',')

# 2. Convertir la matriz en formato de diccionario para QuantumAnnealing.jl
ising_model = Dict()
n = size(qubo_matrix, 1)  # Tamaño de la matriz

for i in 1:n
    for j in i:n  # Solo tomamos la parte superior (simétrica)
        if qubo_matrix[i, j] != 0
            ising_model[(i, j)] = qubo_matrix[i, j]
        end
    end
end

# 3. Ejecutar la simulación de recocido cuántico
annealing_time = 5.0  # Tiempo de recocido (ajustable)
ρ = simulate(ising_model, annealing_time, AS_LINEAR)

# 4. Obtener la solución binaria más probable
solution = argmax(ρ) .- 1  # Convertimos de 1-based a 0/1

# 5. Guardar la solución en solution.csv
writedlm("solution.csv", solution, ',')
println("Solución guardada en solution.csv")