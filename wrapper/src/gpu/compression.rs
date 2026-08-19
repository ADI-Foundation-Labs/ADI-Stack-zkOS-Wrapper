use std::alloc::Global;

use boojum::{
    config::{ProvingCSConfig, SetupCSConfig},
    cs::{
        cs_builder::new_builder,
        cs_builder_reference::CsReferenceImplementationBuilder,
        implementations::{pow::NoPow, prover::ProofConfig, setup::FinalizationHintsForProver},
        traits::circuit::CircuitBuilder,
    },
    field::goldilocks::GoldilocksField,
    worker::Worker,
};
use shivini::{
    ProverContext, ProverContextConfig,
    cs::{GpuSetup, gpu_setup_and_vk_from_base_setup_vk_params_and_hints},
    gpu_proof_config::GpuProofConfig,
    gpu_prove_from_external_witness_data_cancellable,
    prover_stages::{Cancelled, StageTimer},
};

use crate::{
    CompressionCircuit, CompressionProof, CompressionTranscript, CompressionTreeHasher,
    CompressionVK, RiscWrapperProof, RiscWrapperVK, gpu::context::apply_env_overrides,
};

type GL = GoldilocksField;

/// The stage timer is the caller's — see [`crate::gpu::risc_wrapper::get_risc_wrapper_setup`].
pub fn get_compression_setup(
    worker: &Worker,
    risc_wrapper_vk: RiscWrapperVK,
    stages: &mut StageTimer,
) -> Result<
    (
        GpuSetup<CompressionTreeHasher>,
        CompressionVK,
        FinalizationHintsForProver,
    ),
    Cancelled,
> {
    let start = std::time::Instant::now();

    stages.step("compression_setup_gpu_context")?;
    // Currently the GPU context is initialized here, but it should be done at a higher level.
    // For compression circuit, we actually have to set the domain size lower.
    let config = apply_env_overrides(
        ProverContextConfig::default().with_smallest_supported_domain_size(1 << 15),
    );
    let _prover_context = ProverContext::create_with_config(config).unwrap();

    let verify_inner_proof: bool = false;
    let circuit = CompressionCircuit::new(None, risc_wrapper_vk, verify_inner_proof);

    let geometry = CompressionCircuit::geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GL, GL, SetupCSConfig>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GL>(builder_impl);

    let builder = CompressionCircuit::configure_builder(builder);
    let mut cs = builder.build(num_vars.unwrap());

    stages.step("compression_setup_synthesize")?;
    // compression circuit doesn't have any tables.
    circuit.synthesize_into_cs(&mut cs);

    stages.step("compression_setup_pad_and_shrink")?;
    let (_, finalization_hint) = cs.pad_and_shrink();

    let ProofConfig {
        fri_lde_factor,
        merkle_tree_cap_size,
        ..
    } = CompressionCircuit::get_proof_config();
    stages.step("compression_setup_into_assembly")?;
    let cs = cs.into_assembly::<std::alloc::Global>();

    stages.step("compression_setup_light_setup")?;
    let (setup_base, vk_params, vars_hint, witness_hints) =
        cs.get_light_setup(worker, fri_lde_factor, merkle_tree_cap_size);

    stages.step("compression_setup_gpu_setup_and_vk")?;
    let (gpu_setup, gpu_vk) =
        gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<CompressionTreeHasher, _>(
            setup_base.clone(),
            vk_params,
            vars_hint.clone(),
            witness_hints.clone(),
            &worker,
        )
        .unwrap();

    println!(
        "compression circuit setup takes {} ms",
        start.elapsed().as_millis()
    );

    Ok((gpu_setup, gpu_vk, finalization_hint))
}

/// The stage timer is the caller's — see [`get_compression_setup`].
pub fn prove_compression(
    risc_wrapper_proof: RiscWrapperProof,
    risc_wrapper_vk: RiscWrapperVK,
    finalization_hint: &FinalizationHintsForProver,
    gpu_setup: &GpuSetup<CompressionTreeHasher>,
    gpu_vk: &CompressionVK,
    worker: &Worker,
    stages: &mut StageTimer,
) -> Result<CompressionProof, Cancelled> {
    let start = std::time::Instant::now();

    stages.step("compression_prove_gpu_context")?;
    // Currently the GPU context is initialized here, but it should be done at a higher level.
    // For compression circuit, we actually have to set the domain size lower.
    let config = apply_env_overrides(
        ProverContextConfig::default().with_smallest_supported_domain_size(1 << 15),
    );
    let _prover_context = ProverContext::create_with_config(config).unwrap();

    let verify_inner_proof = true;
    let circuit = CompressionCircuit::new(
        Some(risc_wrapper_proof),
        risc_wrapper_vk,
        verify_inner_proof,
    );

    let geometry = CompressionCircuit::geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GL, GL, ProvingCSConfig>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GL>(builder_impl);

    let builder = CompressionCircuit::configure_builder(builder);
    let mut cs = builder.build(num_vars.unwrap());

    stages.step("compression_prove_synthesize")?;
    // compression circuit doesn't have any tables.
    circuit.synthesize_into_cs(&mut cs);

    stages.step("compression_prove_pad_and_shrink")?;
    cs.pad_and_shrink_using_hint(finalization_hint);

    stages.step("compression_prove_into_assembly")?;
    let cs = cs.into_assembly::<std::alloc::Global>();

    // shivini opens its own timeline inside the call below — see `risc_wrapper::prove_risc_wrapper`.
    stages.step("compression_prove_gpu")?;
    let gpu_proof_config = GpuProofConfig::from_assembly(&cs);

    let external_witness_data = cs.witness.unwrap();

    let proof_config = CompressionCircuit::get_proof_config();

    let proof = gpu_prove_from_external_witness_data_cancellable::<
        CompressionTranscript,
        CompressionTreeHasher,
        NoPow,
        Global,
    >(
        &gpu_proof_config,
        &external_witness_data,
        proof_config,
        &gpu_setup,
        &gpu_vk,
        (),
        worker,
    )
    .unwrap()
    .ok_or(Cancelled {
        stage: "compression_prove_gpu",
    })?;

    println!(
        "compression wrapper proving takes {} ms",
        start.elapsed().as_millis()
    );

    Ok(proof.into())
}
