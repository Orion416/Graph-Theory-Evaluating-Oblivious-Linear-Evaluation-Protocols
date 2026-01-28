# -*- coding: utf-8 -*-
"""Protocol Definitions for OLE Communication Graph Analysis"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from enum import Enum


class CryptoFoundation(Enum):
    """Cryptographic foundation categories for OLE protocols."""
    HOMOMORPHIC_ENCRYPTION = "HE"
    OT_EXTENSION = "OT"
    ALGEBRAIC_STRUCTURE = "Algebraic"
    SUBLINEAR_COMMUNICATION = "Sublinear"
    BASELINE_OT = "Baseline"


@dataclass
class SecurityParameters:
    """Security parameters for cryptographic protocols."""
    security_level: int = 128  # bits
    field_size: int = 256  # bits

    @property
    def paillier_modulus_bits(self) -> int:
        """Paillier modulus size: typically 16x security level for 128-bit security."""
        return self.security_level * 16  # 2048 bits for 128-bit security

    @property
    def group_element_bits(self) -> int:
        """Group element size for DDH-based protocols."""
        return self.security_level * 4  # 512 bits

    @property
    def ot_base_bits(self) -> int:
        """Base OT message size."""
        return self.security_level * 2  # 256 bits

    @property
    def lattice_dimension(self) -> int:
        """LWE lattice dimension for post-quantum security."""
        return self.security_level * 4  # n = 512

    @property
    def lattice_modulus_bits(self) -> int:
        """LWE modulus bits (log q)."""
        return 32  # q = 2^32 for practical implementations

    @property
    def lattice_element_size(self) -> int:
        """Size of a lattice element in bits."""
        # n * log2(q) bits, but compressed
        return self.lattice_dimension * self.lattice_modulus_bits // 8


@dataclass
class ComputationCost:
    """Computation cost description."""
    # Symmetric crypto ops
    hash_ops: int = 0          # Hash ops
    aes_ops: int = 0           # AES ops
    prg_ops: int = 0           # PRG ops

    # Modular arithmetic ops
    ntt_ops: int = 0           # NTT ops
    mod_mul_ops: int = 0       # Modular multiplication ops

    # Elliptic curve ops
    ec_mul_ops: int = 0        # Scalar multiplication ops
    ec_add_ops: int = 0        # Point addition ops

    # Homomorphic encryption ops
    he_enc_ops: int = 0        # HE encryption ops
    he_mul_ops: int = 0        # HE multiplication ops

    def to_aes_equivalent(self) -> int:
        """Convert all ops to AES equivalent."""
        # Conversion coefficients based on benchmarks
        # eBASH benchmark: SHA-256 ≈ 100 AES
        # SEAL guide: NTT butterfly ≈ 0.5 AES
        # Curve25519 paper: EC scalar mul ≈ 2000 AES
        # OpenFHE benchmark: HE ciphertext mul ≈ 5000 AES
        coefficients = {
            'hash': 100,
            'aes': 1,
            'prg': 10,
            'ntt': 0.5,
            'mod_mul': 1,
            'ec_mul': 2000,
            'ec_add': 50,
            'he_enc': 3000,
            'he_mul': 5000,
        }

        total = (
            self.hash_ops * coefficients['hash'] +
            self.aes_ops * coefficients['aes'] +
            self.prg_ops * coefficients['prg'] +
            self.ntt_ops * coefficients['ntt'] +
            self.mod_mul_ops * coefficients['mod_mul'] +
            self.ec_mul_ops * coefficients['ec_mul'] +
            self.ec_add_ops * coefficients['ec_add'] +
            self.he_enc_ops * coefficients['he_enc'] +
            self.he_mul_ops * coefficients['he_mul']
        )
        return int(total)


@dataclass
class Message:
    """Represents a single message in protocol communication."""
    sender: int
    receiver: int
    round_num: int
    size_bits: int
    description: str = ""
    computation: ComputationCost = field(default_factory=ComputationCost)


@dataclass
class Protocol:
    """Represents an OLE protocol with its communication pattern."""
    identifier: str
    name: str
    foundation: CryptoFoundation
    num_parties: int
    messages: List[Message] = field(default_factory=list)

    @property
    def num_rounds(self) -> int:
        """Compute the number of communication rounds from messages."""
        if not self.messages:
            return 0
        return max(msg.round_num for msg in self.messages)

    @property
    def num_edges(self) -> int:
        """Number of directed edges (messages) in the communication graph."""
        return len(self.messages)

    @property
    def total_edge_weight(self) -> int:
        """Total communication in bits."""
        return sum(msg.size_bits for msg in self.messages)

    @property
    def average_edge_weight(self) -> float:
        """Average message size in bits."""
        if not self.messages:
            return 0.0
        return self.total_edge_weight / self.num_edges

    @property
    def total_computation_cost(self) -> int:
        """Total protocol computation cost (AES ops)."""
        return sum(msg.computation.to_aes_equivalent() for msg in self.messages)

    @property
    def computation_by_party(self) -> Tuple[int, int]:
        """Computation cost by party."""
        party_0_cost = sum(
            msg.computation.to_aes_equivalent()
            for msg in self.messages if msg.sender == 0
        )
        party_1_cost = sum(
            msg.computation.to_aes_equivalent()
            for msg in self.messages if msg.sender == 1
        )
        return party_0_cost, party_1_cost


class ProtocolFactory:
    """Factory class for creating protocol definitions based on security parameters."""

    def __init__(self, params: SecurityParameters = None):
        self.params = params or SecurityParameters()

    def create_paillier_ole(self) -> Protocol:
        """Create Paillier-OLE protocol definition."""
        p = self.params
        # Paillier ciphertext size
        ciphertext_size = p.paillier_modulus_bits  # 2048 bits

        # Computation params - based on Paillier ops
        # Each encryption needs ~2 mod exp, each mod exp ~log2(n) mod mul
        modulus_bits = p.paillier_modulus_bits
        mod_muls_per_exp = modulus_bits  # ~2048 mod muls

        messages = [
            # Round 1: Setup - Key gen requires hash and mod
            Message(0, 1, 1, ciphertext_size, "Public key A->B",
                   ComputationCost(hash_ops=2, mod_mul_ops=modulus_bits // 4)),
            # Round 2: Main computation
            # B enc input requires HE enc
            Message(1, 0, 2, ciphertext_size + ciphertext_size // 2, "Encrypted input B->A",
                   ComputationCost(he_enc_ops=1, mod_mul_ops=mod_muls_per_exp * 2)),
            # A calc result requires HE mul
            Message(0, 1, 2, ciphertext_size, "Encrypted result A->B",
                   ComputationCost(he_mul_ops=1, mod_mul_ops=mod_muls_per_exp)),
            # Round 3: Verification - Verification requires decryption
            Message(1, 0, 3, ciphertext_size + ciphertext_size // 2, "Verification B->A",
                   ComputationCost(he_enc_ops=1, mod_mul_ops=mod_muls_per_exp * 2, hash_ops=1)),
        ]

        return Protocol(
            identifier="P1",
            name="Paillier-OLE",
            foundation=CryptoFoundation.HOMOMORPHIC_ENCRYPTION,
            num_parties=2,
            messages=messages
        )

    def create_lwe_ole(self) -> Protocol:
        """Create LWE-OLE protocol definition."""
        p = self.params
        # Lattice element size: n * log(q) / 4 for compressed representation
        lattice_elem = p.lattice_element_size  # 2048 bits
        n = p.lattice_dimension  # 512

        # NTT ops: n * log2(n) butterflies
        log_n = int(np.log2(n))
        ntt_butterflies = n * log_n

        size_a_to_b = lattice_elem + lattice_elem // 2 + lattice_elem // 4  # 3584
        size_b_to_a = 2 * lattice_elem + lattice_elem // 2  # 5120

        messages = [
            # Round 1: Matrix and masked inputs
            # Public matrix requires NTT
            Message(0, 1, 1, size_a_to_b, "Public matrix A",
                   ComputationCost(ntt_ops=ntt_butterflies * 2, mod_mul_ops=n * 4, hash_ops=4)),
            # Masked input requires NTT and mod mul
            Message(1, 0, 1, size_b_to_a, "Masked input vector",
                   ComputationCost(ntt_ops=ntt_butterflies * 3, mod_mul_ops=n * 6)),
            # Round 2: Results
            # Encrypted linear comb requires NTT and mod mul
            Message(0, 1, 2, size_a_to_b - lattice_elem // 4, "Encrypted linear combination",
                   ComputationCost(ntt_ops=ntt_butterflies * 2, mod_mul_ops=n * 4)),
            # Decryption shares require NTT and mod mul
            Message(1, 0, 2, lattice_elem, "Decryption shares",
                   ComputationCost(ntt_ops=ntt_butterflies, mod_mul_ops=n * 2)),
        ]

        return Protocol(
            identifier="P2",
            name="LWE-OLE",
            foundation=CryptoFoundation.HOMOMORPHIC_ENCRYPTION,
            num_parties=2,
            messages=messages
        )

    def create_ddh_ole(self) -> Protocol:
        """Create DDH-OLE protocol definition."""
        p = self.params
        group_elem = p.group_element_bits  # 512 bits

        # Ops per message
        messages = [
            # Round 1: Masked inputs (symmetric)
            Message(0, 1, 1, group_elem, "Masked input A->B",
                   ComputationCost(ec_mul_ops=1, hash_ops=1)),
            Message(1, 0, 1, group_elem, "Masked input B->A",
                   ComputationCost(ec_mul_ops=1, hash_ops=1)),
            # Round 2: Blinded results (symmetric)
            Message(0, 1, 2, group_elem, "Blinded result A->B",
                   ComputationCost(ec_mul_ops=2, ec_add_ops=1)),
            Message(1, 0, 2, group_elem, "Blinded result B->A",
                   ComputationCost(ec_mul_ops=2, ec_add_ops=1)),
        ]

        return Protocol(
            identifier="P3",
            name="DDH-OLE",
            foundation=CryptoFoundation.ALGEBRAIC_STRUCTURE,
            num_parties=2,
            messages=messages
        )

    def create_mascot_ole(self) -> Protocol:
        """Create MASCOT-OLE protocol definition."""
        p = self.params
        kappa = p.security_level  # 128

        # Design for high concentration (unequal weights)
        base_small = kappa * 2  # 256
        base_med = kappa * 3    # 384
        base_large = kappa * 5  # 640

        # OT extension ops per OT
        ot_hash_per_msg = kappa // 4
        prg_per_msg = kappa // 8

        messages = [
            # Round 1: Base OT (2 messages)
            Message(0, 1, 1, base_small, "Base OT setup A->B",
                   ComputationCost(hash_ops=ot_hash_per_msg, aes_ops=kappa)),
            Message(1, 0, 1, base_med, "Base OT response B->A",
                   ComputationCost(hash_ops=ot_hash_per_msg, aes_ops=kappa)),
            # Round 2: OT Extension (3 messages)
            Message(0, 1, 2, base_med, "OT extension A->B",
                   ComputationCost(hash_ops=ot_hash_per_msg * 2, prg_ops=prg_per_msg)),
            Message(1, 0, 2, base_large, "OT extension B->A",
                   ComputationCost(hash_ops=ot_hash_per_msg * 3, prg_ops=prg_per_msg * 2)),
            Message(0, 1, 2, base_small // 2, "OT extension cont A->B",
                   ComputationCost(hash_ops=ot_hash_per_msg)),
            # Round 3: Correlation check (2 messages)
            Message(0, 1, 3, base_med, "Correlation check A->B",
                   ComputationCost(hash_ops=ot_hash_per_msg * 2)),
            Message(1, 0, 3, base_med, "Correlation check B->A",
                   ComputationCost(hash_ops=ot_hash_per_msg * 2)),
            # Round 4: Input sharing (2 messages)
            Message(0, 1, 4, base_small, "Input share A->B",
                   ComputationCost(hash_ops=ot_hash_per_msg, aes_ops=kappa // 2)),
            Message(1, 0, 4, base_large, "Input share B->A",
                   ComputationCost(hash_ops=ot_hash_per_msg * 2, aes_ops=kappa)),
            # Round 5: Output (1 message)
            Message(1, 0, 5, base_med, "Output reconstruction B->A",
                   ComputationCost(hash_ops=ot_hash_per_msg)),
        ]

        return Protocol(
            identifier="P4",
            name="MASCOT-OLE",
            foundation=CryptoFoundation.OT_EXTENSION,
            num_parties=2,
            messages=messages
        )

    def create_spdz_ole(self) -> Protocol:
        """Create SPDZ-OLE protocol definition."""
        p = self.params
        kappa = p.security_level

        preproc_large = kappa * 8   # 1024
        preproc_small = kappa * 6   # 768
        online_large = kappa * 5    # 640
        online_small = kappa * 3    # 384

        # Preprocessing involves OT
        preproc_hash = kappa // 2
        preproc_aes = kappa

        messages = [
            # Preprocessing phase (rounds 1-2)
            Message(0, 1, 1, preproc_small, "Preprocessing A->B (1)",
                   ComputationCost(hash_ops=preproc_hash, aes_ops=preproc_aes, prg_ops=kappa // 4)),
            Message(1, 0, 1, preproc_large, "Preprocessing B->A (1)",
                   ComputationCost(hash_ops=preproc_hash * 2, aes_ops=preproc_aes * 2)),
            Message(0, 1, 2, preproc_small, "Preprocessing A->B (2)",
                   ComputationCost(hash_ops=preproc_hash, prg_ops=kappa // 4)),
            Message(1, 0, 2, preproc_large, "Preprocessing B->A (2)",
                   ComputationCost(hash_ops=preproc_hash * 2, aes_ops=preproc_aes)),
            # Online phase (rounds 3-4) - Online phase: mainly Hash
            Message(0, 1, 3, online_small, "Online A->B (1)",
                   ComputationCost(hash_ops=kappa // 4)),
            Message(1, 0, 3, online_large, "Online B->A (1)",
                   ComputationCost(hash_ops=kappa // 2)),
            Message(0, 1, 4, online_small, "Online A->B (2)",
                   ComputationCost(hash_ops=kappa // 4)),
            Message(1, 0, 4, online_large, "Online B->A (2)",
                   ComputationCost(hash_ops=kappa // 2)),
        ]

        return Protocol(
            identifier="P5",
            name="SPDZ-OLE",
            foundation=CryptoFoundation.OT_EXTENSION,
            num_parties=2,
            messages=messages
        )

    def create_fss_ole(self) -> Protocol:
        """Create FSS-OLE protocol definition."""
        p = self.params
        kappa = p.security_level

        share_large = kappa * 6   # 768
        share_small = kappa * 4   # 512
        eval_size = kappa * 2     # 256
        recon_size = kappa * 3    # 384

        # FSS main ops: PRG and Hash
        prg_per_share = kappa // 2
        hash_per_eval = kappa // 4

        messages = [
            # Round 1: Share distribution (dealer v1 sends larger shares)
            Message(0, 1, 1, share_large, "Function share A->B (1)",
                   ComputationCost(prg_ops=prg_per_share * 2, hash_ops=hash_per_eval)),
            Message(0, 1, 1, share_small, "Function share A->B (2)",
                   ComputationCost(prg_ops=prg_per_share, hash_ops=hash_per_eval)),
            # Round 2: Evaluation (v2 returns results)
            Message(1, 0, 2, share_large, "Evaluation result B->A (1)",
                   ComputationCost(prg_ops=prg_per_share * 2, aes_ops=kappa)),
            Message(1, 0, 2, eval_size, "Evaluation result B->A (2)",
                   ComputationCost(prg_ops=prg_per_share, hash_ops=hash_per_eval)),
            # Round 3: Reconstruction
            Message(0, 1, 3, recon_size, "Reconstruction A->B",
                   ComputationCost(hash_ops=hash_per_eval * 2)),
            Message(1, 0, 3, share_small, "Reconstruction B->A",
                   ComputationCost(hash_ops=hash_per_eval)),
        ]

        return Protocol(
            identifier="P6",
            name="FSS-OLE",
            foundation=CryptoFoundation.ALGEBRAIC_STRUCTURE,
            num_parties=2,
            messages=messages
        )

    def create_ferret_ole(self) -> Protocol:
        """Create Ferret-OLE protocol definition."""
        p = self.params
        kappa = p.security_level

        seed_large = kappa * 5   # 640
        seed_small = kappa * 3   # 384
        expand_size = kappa * 3  # 384
        eval_size = kappa * 2    # 256

        # Ferret core: PRG expansion
        prg_expand = kappa
        aes_per_msg = kappa // 2

        messages = [
            # Round 1: Seed exchange (heavy)
            Message(0, 1, 1, seed_small, "Seed A->B",
                   ComputationCost(prg_ops=prg_expand, hash_ops=kappa // 8)),
            Message(1, 0, 1, seed_large, "Seed B->A",
                   ComputationCost(prg_ops=prg_expand * 2, hash_ops=kappa // 4)),
            # Round 2: Expansion
            Message(0, 1, 2, expand_size, "Expansion A->B",
                   ComputationCost(prg_ops=prg_expand * 2, aes_ops=aes_per_msg)),
            Message(1, 0, 2, seed_large, "Expansion B->A",
                   ComputationCost(prg_ops=prg_expand * 3, aes_ops=aes_per_msg * 2)),
            # Round 3: Evaluation (light)
            Message(0, 1, 3, eval_size, "Evaluation A->B",
                   ComputationCost(hash_ops=kappa // 8)),
            Message(1, 0, 3, expand_size, "Evaluation B->A",
                   ComputationCost(hash_ops=kappa // 4)),
        ]

        return Protocol(
            identifier="P7",
            name="Ferret-OLE",
            foundation=CryptoFoundation.OT_EXTENSION,
            num_parties=2,
            messages=messages
        )

    def create_silent_ole(self) -> Protocol:
        """Create Silent-OLE protocol definition."""
        p = self.params
        kappa = p.security_level

        boot_large = kappa * 4  # 512
        boot_small = kappa * 2  # 256
        expand_large = kappa * 3  # 384
        expand_small = kappa * 3  # 384

        # Main cost: local PRG expansion
        # Seed expansion requires PRG
        prg_bootstrap = kappa * 2
        prg_expand = kappa * 4

        messages = [
            # Round 1: Bootstrap (heavier computation)
            Message(0, 1, 1, boot_small, "Bootstrap A->B",
                   ComputationCost(prg_ops=prg_bootstrap, hash_ops=kappa // 4, aes_ops=kappa)),
            Message(1, 0, 1, boot_large, "Bootstrap B->A",
                   ComputationCost(prg_ops=prg_bootstrap * 2, hash_ops=kappa // 2)),
            # Round 2: Expansion (heavy local computation)
            Message(0, 1, 2, expand_small, "Expansion A->B",
                   ComputationCost(prg_ops=prg_expand, aes_ops=kappa * 2)),
            Message(1, 0, 2, expand_large, "Expansion B->A",
                   ComputationCost(prg_ops=prg_expand * 2, aes_ops=kappa * 2)),
        ]

        return Protocol(
            identifier="P8",
            name="Silent-OLE",
            foundation=CryptoFoundation.SUBLINEAR_COMMUNICATION,
            num_parties=2,
            messages=messages
        )

    def create_np_ot(self) -> Protocol:
        """Create Naor-Pinkas OT protocol definition (baseline)."""
        p = self.params
        group_elem = p.field_size  # 256 bits

        messages = [
            # Round 1: Setup (symmetric)
            Message(0, 1, 1, group_elem, "Setup A->B",
                   ComputationCost(ec_mul_ops=1, hash_ops=1)),
            Message(1, 0, 1, group_elem, "Choice B->A",
                   ComputationCost(ec_mul_ops=1, ec_add_ops=1)),
            # Round 2: Transfer (symmetric)
            Message(0, 1, 2, group_elem, "Transfer A->B",
                   ComputationCost(ec_mul_ops=2, hash_ops=2)),
            Message(1, 0, 2, group_elem, "Response B->A",
                   ComputationCost(ec_mul_ops=1, hash_ops=1)),
        ]

        return Protocol(
            identifier="P0",
            name="NP-OT",
            foundation=CryptoFoundation.BASELINE_OT,
            num_parties=2,
            messages=messages
        )

    def create_all_protocols(self) -> Dict[str, Protocol]:
        """Create all protocols and return as dictionary."""
        protocols = {}

        protocols["P0"] = self.create_np_ot()
        protocols["P1"] = self.create_paillier_ole()
        protocols["P2"] = self.create_lwe_ole()
        protocols["P3"] = self.create_ddh_ole()
        protocols["P4"] = self.create_mascot_ole()
        protocols["P5"] = self.create_spdz_ole()
        protocols["P6"] = self.create_fss_ole()
        protocols["P7"] = self.create_ferret_ole()
        protocols["P8"] = self.create_silent_ole()

        return protocols


def get_protocols(params: SecurityParameters = None) -> Dict[str, Protocol]:
    """Convenience function to get all protocols with specified parameters."""
    factory = ProtocolFactory(params)
    return factory.create_all_protocols()


if __name__ == "__main__":
    # Test protocol creation
    params = SecurityParameters(security_level=128, field_size=256)
    protocols = get_protocols(params)

    print("Protocol Communication Summary")
    print("=" * 80)
    print(f"{'ID':<4} {'Name':<15} {'|V|':<4} {'|E|':<4} {'Total Weight':<14} {'Rounds':<7} {'Avg Weight':<12}")
    print("-" * 80)

    for pid in ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]:
        p = protocols[pid]
        print(f"{p.identifier:<4} {p.name:<15} {p.num_parties:<4} {p.num_edges:<4} "
              f"{p.total_edge_weight:<14} {p.num_rounds:<7} {p.average_edge_weight:<12.1f}")
