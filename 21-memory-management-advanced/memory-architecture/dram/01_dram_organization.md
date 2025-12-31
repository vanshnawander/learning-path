# DRAM Organization: Deep Dive

## Physical Structure of DRAM

### Hierarchical Organization

Modern DRAM is organized hierarchically:

```
Channel → DIMM → Rank → Chip → Bank → Row → Column
```

**Channel**: Independent memory controller connection
- Each channel has its own 64-bit data bus
- Dual-channel doubles bandwidth (2x64 = 128 bits)
- Modern CPUs: 2-8 channels (server CPUs up to 12)

**DIMM (Dual Inline Memory Module)**: Physical module
- Contains multiple DRAM chips
- UDIMM (unbuffered), RDIMM (registered), LRDIMM (load-reduced)

**Rank**: Set of chips that respond together
- Single-rank, dual-rank, quad-rank DIMMs
- More ranks = more parallelism but higher latency

**Bank**: Independent memory array within chip
- DDR4: 16 banks (4 bank groups × 4 banks)
- DDR5: 32 banks (8 bank groups × 4 banks)
- Banks can be accessed in parallel

**Row (Page)**: Horizontal line of cells
- Typical size: 8KB-16KB
- Activated into row buffer for access
- Row hit vs row miss dramatically affects latency

**Column**: Actual data location
- Burst access: 8 consecutive columns (DDR4/5)

### DRAM Cell Operation

```
        Word Line (Row Select)
             |
             v
        +----+----+
        |    |    |
        | C  | T  |  C = Capacitor (stores bit)
        |    |    |  T = Access Transistor
        +----+----+
             |
             v
        Bit Line (Data)
```

**Read Operation**:
1. Precharge bit line to Vdd/2
2. Assert word line (opens transistor)
3. Charge sharing between capacitor and bit line
4. Sense amplifier detects tiny voltage difference
5. Amplify to full logic level
6. **Destructive read**: must write back

**Refresh Requirement**:
- Capacitor leaks charge over time
- Must refresh every ~64ms (DDR4)
- DDR5: Same window but distributed differently
- Refresh steals bandwidth (typically 5-10%)

### Timing Parameters (DDR4-3200 Example)

| Parameter | Symbol | Cycles | Time (ns) |
|-----------|--------|--------|-----------|
| CAS Latency | CL | 22 | 13.75 |
| RAS to CAS | tRCD | 22 | 13.75 |
| Row Precharge | tRP | 22 | 13.75 |
| Row Active Time | tRAS | 52 | 32.5 |
| Row Cycle Time | tRC | 74 | 46.25 |

**Access Patterns**:

```
Row Hit (Best): ~13ns
  - Data already in row buffer
  - Just column access needed

Row Miss (Empty): ~27ns  
  - Activate row, then column access
  - tRCD + CL

Row Conflict (Worst): ~40ns
  - Precharge current row
  - Activate new row
  - Column access
  - tRP + tRCD + CL
```

### Bank Interleaving

Memory controllers interleave addresses across banks to hide latency:

```c
// Physical address mapping (simplified)
// Address bits distributed across:
// [Row][Bank Group][Bank][Column][Channel][Byte Offset]

// Example: 64-byte cache line access
// Channel bits: 1 (dual channel)
// Bank bits: 4 (16 banks)
// Consecutive cache lines go to different banks
```

**XOR-based Interleaving** (used by Intel/AMD):
```
Effective Bank = Bank_bits XOR (Row_bits[low])
```
This reduces row conflicts for sequential access patterns.

## DDR4 vs DDR5 Comparison

| Feature | DDR4 | DDR5 |
|---------|------|------|
| Density | Up to 16Gb/chip | Up to 64Gb/chip |
| Voltage | 1.2V | 1.1V |
| Prefetch | 8n | 16n |
| Burst Length | 8 | 16 |
| Bank Groups | 4 | 8 |
| Banks/Group | 4 | 4 |
| Total Banks | 16 | 32 |
| Channels/DIMM | 1 | 2 (32-bit each) |
| ECC | Optional (ECC DIMMs) | On-die ECC standard |
| Max Speed | 3200 MT/s | 8800+ MT/s |

### DDR5 Key Improvements

1. **Two Independent Channels per DIMM**
   - Each 32-bit wide (vs single 64-bit)
   - Better fine-grained parallelism
   - Better bandwidth utilization

2. **On-Die ECC**
   - Corrects single-bit errors within chip
   - Improves reliability at high densities
   - Note: Not same as system ECC

3. **Same Bank Refresh**
   - Only refreshes banks in same bank group
   - Other bank groups remain accessible
   - Better sustained bandwidth

## Industry Best Practices

### Server Memory Configuration

**For Maximum Bandwidth**:
- Populate all channels equally
- Use single-rank DIMMs if latency-critical
- Match DIMM speeds across all slots

**For Maximum Capacity**:
- Use LRDIMM for highest density
- Accept some latency penalty
- Consider quad-rank DIMMs

### Memory Subsystem Tuning (Linux)

```bash
# Check current NUMA topology
numactl --hardware

# Check memory configuration
dmidecode -t memory | grep -E "Size|Speed|Type"

# View memory controller stats (Intel)
sudo pcm-memory

# View DIMM thermal throttling
sudo ipmitool sdr type memory
```

### Application Considerations

1. **Streaming Access**: 
   - Sequential access maximizes row hits
   - Prefetchers work well
   - ~90% of peak bandwidth achievable

2. **Random Access**:
   - Row conflicts destroy performance
   - May achieve only 10-20% of peak bandwidth
   - Consider data layout changes

3. **Pointer Chasing**:
   - Worst case for DRAM
   - Each access depends on previous
   - Full latency on critical path

## References

- JEDEC DDR4/DDR5 Specifications
- "Memory Systems: Cache, DRAM, Disk" - Jacob, Ng, Wang
- Intel/AMD Memory Controller Documentation
