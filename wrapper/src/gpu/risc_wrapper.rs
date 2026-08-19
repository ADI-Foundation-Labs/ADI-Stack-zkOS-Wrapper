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
    ProverContext,
    cs::{GpuSetup, gpu_setup_and_vk_from_base_setup_vk_params_and_hints},
    gpu_proof_config::GpuProofConfig,
    gpu_prove_from_external_witness_data_cancellable,
    prover_stages::{Cancelled, StageTimer},
};

use crate::{
    BinaryCommitment, RiscWrapper, RiscWrapperProof, RiscWrapperTranscript, RiscWrapperTreeHasher,
    RiscWrapperVK, RiscWrapperWitness, gpu::context::build_prover_context_config,
};

type GL = GoldilocksField;

/// The stage timer is the caller's, so a cancel arriving between phases is not lost to a
/// fresh baseline — see [`crate::prove_cancellable`].
pub fn get_risc_wrapper_setup(
    worker: &Worker,
    binary_commitment: BinaryCommitment,
    stages: &mut StageTimer,
) -> Result<
    (
        GpuSetup<RiscWrapperTreeHasher>,
        RiscWrapperVK,
        FinalizationHintsForProver,
    ),
    Cancelled,
> {
    let start = std::time::Instant::now();

    stages.step("setup_gpu_context")?;
    // Currently the GPU context is initialized here, but it should be done at a higher level.
    let _prover_context = ProverContext::create_with_config(build_prover_context_config()).unwrap();

    let verify_inner_proof: bool = false;
    let circuit = RiscWrapper::new(None, verify_inner_proof, binary_commitment);

    let geometry = RiscWrapper::geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GL, GL, SetupCSConfig>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GL>(builder_impl);

    let builder = RiscWrapper::configure_builder(builder);
    let mut cs = builder.build(num_vars.unwrap());
    circuit.add_tables(&mut cs);

    stages.step("setup_synthesize")?;
    circuit.synthesize_into_cs(&mut cs);

    stages.step("setup_pad_and_shrink")?;
    let (_, finalization_hint) = cs.pad_and_shrink();

    let ProofConfig {
        fri_lde_factor,
        merkle_tree_cap_size,
        ..
    } = RiscWrapper::get_proof_config();
    stages.step("setup_into_assembly")?;
    let cs = cs.into_assembly::<std::alloc::Global>();

    stages.step("setup_light_setup")?;
    let (setup_base, vk_params, vars_hint, witness_hints) =
        cs.get_light_setup(worker, fri_lde_factor, merkle_tree_cap_size);

    stages.step("setup_gpu_setup_and_vk")?;
    let (gpu_setup, gpu_vk) =
        gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<RiscWrapperTreeHasher, _>(
            setup_base.clone(),
            vk_params,
            vars_hint.clone(),
            witness_hints.clone(),
            &worker,
        )
        .unwrap();

    println!(
        "risc wrapper setup takes {} ms",
        start.elapsed().as_millis()
    );

    Ok((gpu_setup, gpu_vk, finalization_hint))
}

/// The stage timer is the caller's — see [`get_risc_wrapper_setup`].
pub fn prove_risc_wrapper(
    risc_wrapper_witness: RiscWrapperWitness,
    finalization_hint: &FinalizationHintsForProver,
    gpu_setup: &GpuSetup<RiscWrapperTreeHasher>,
    gpu_vk: &RiscWrapperVK,
    worker: &Worker,
    binary_commitment: BinaryCommitment,
    stages: &mut StageTimer,
) -> Result<RiscWrapperProof, Cancelled> {
    let start = std::time::Instant::now();

    stages.step("prove_gpu_context")?;
    // Currently the GPU context is initialized here, but it should be done at a higher level.
    let _prover_context = ProverContext::create_with_config(build_prover_context_config()).unwrap();

    let verify_inner_proof = true;
    let circuit = RiscWrapper::new(
        Some(risc_wrapper_witness),
        verify_inner_proof,
        binary_commitment,
    );

    let geometry = RiscWrapper::geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GL, GL, ProvingCSConfig>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GL>(builder_impl);

    let builder = RiscWrapper::configure_builder(builder);
    let mut cs = builder.build(num_vars.unwrap());
    circuit.add_tables(&mut cs);

    stages.step("prove_synthesize")?;
    circuit.synthesize_into_cs(&mut cs);

    stages.step("prove_pad_and_shrink")?;
    cs.pad_and_shrink_using_hint(finalization_hint);

    stages.step("prove_into_assembly")?;
    let cs = cs.into_assembly::<std::alloc::Global>();

    // shivini opens its own timeline inside the call below, so this stage's duration is the
    // sum of the stages it reports. One timeline per proof needs a `&mut StageTimer` on
    // shivini's public entry point.
    stages.step("prove_gpu")?;
    let gpu_proof_config = GpuProofConfig::from_assembly(&cs);

    let external_witness_data = cs.witness.unwrap();

    let proof_config = RiscWrapper::get_proof_config();

    let proof = gpu_prove_from_external_witness_data_cancellable::<
        RiscWrapperTranscript,
        RiscWrapperTreeHasher,
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
    .ok_or(Cancelled { stage: "prove_gpu" })?;

    println!(
        "risc wrapper proving takes {} ms",
        start.elapsed().as_millis()
    );

    Ok(proof.into())
}
