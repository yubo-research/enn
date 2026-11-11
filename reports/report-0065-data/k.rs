use std::time::Instant;
#[inline(always)]
fn body(a:&[f32],b:&[f32])->f32{const L:usize=8;let mut acc=[0.0f32;L];let(ac,ar)=a.as_chunks::<L>();let(bc,br)=b.as_chunks::<L>();
for(ca,cb)in ac.iter().zip(bc.iter()){for((l,&x),&y)in acc.iter_mut().zip(ca).zip(cb){let d=x-y;*l+=d*d;}}
let mut s:f32=acc.iter().sum();for(&x,&y)in ar.iter().zip(br.iter()){let d=x-y;s+=d*d;}s}
#[target_feature(enable="avx2")] unsafe fn old(a:&[f32],b:&[f32])->f32{body(a,b)}
#[inline(always)]
fn body4(a:&[f32],b:&[f32])->f32{const L:usize=32;let mut acc=[0.0f32;L];let(ac,ar)=a.as_chunks::<L>();let(bc,br)=b.as_chunks::<L>();
for(ca,cb)in ac.iter().zip(bc.iter()){for((l,&x),&y)in acc.iter_mut().zip(ca).zip(cb){let d=x-y;*l+=d*d;}}
let mut s:f32=acc.iter().sum();for(&x,&y)in ar.iter().zip(br.iter()){let d=x-y;s+=d*d;}s}
#[target_feature(enable="avx2")] unsafe fn new4(a:&[f32],b:&[f32])->f32{body4(a,b)}
#[target_feature(enable="avx2,fma")] unsafe fn new4f(a:&[f32],b:&[f32])->f32{
 let (ac,ar)=a.as_chunks::<32>();let(bc,br)=b.as_chunks::<32>();let mut acc=[0.0f32;32];
 for(ca,cb)in ac.iter().zip(bc.iter()){for((l,&x),&y)in acc.iter_mut().zip(ca).zip(cb){let d=x-y;*l=d.mul_add(d,*l);}}
 let mut s:f32=acc.iter().sum();for(&x,&y)in ar.iter().zip(br.iter()){let d=x-y;s+=d*d;}s}
fn main(){let d=1000;let rows:usize=std::env::args().nth(1).unwrap().parse().unwrap();let q:Vec<f32>=(0..d).map(|i|(i as f32*0.37).sin()).collect();
let data:Vec<f32>=(0..rows*d).map(|i|(i as f32*0.011).cos()).collect();
for (name,f) in [("old",old as unsafe fn(&[f32],&[f32])->f32),("acc32",new4),("acc32fma",new4f),("intr4",intr)]{
 let mut tot=0.0;let t=Instant::now();let reps=2_000_000/rows;
 for _ in 0..reps{for r in 0..rows{tot+=unsafe{f(&q,&data[r*d..(r+1)*d])};}}
 let el=t.elapsed().as_secs_f64();println!("{name}: {:.1} ns/row  ({tot:e})",el/(reps*rows) as f64*1e9);}}
use std::arch::x86_64::*;
#[target_feature(enable="avx2,fma")] unsafe fn intr(a:&[f32],b:&[f32])->f32{
 let n=a.len();let mut i=0;let (pa,pb)=(a.as_ptr(),b.as_ptr());
 let mut s0=_mm256_setzero_ps();let mut s1=s0;let mut s2=s0;let mut s3=s0;
 while i+32<=n{
  let d0=_mm256_sub_ps(_mm256_loadu_ps(pa.add(i)),_mm256_loadu_ps(pb.add(i)));s0=_mm256_fmadd_ps(d0,d0,s0);
  let d1=_mm256_sub_ps(_mm256_loadu_ps(pa.add(i+8)),_mm256_loadu_ps(pb.add(i+8)));s1=_mm256_fmadd_ps(d1,d1,s1);
  let d2=_mm256_sub_ps(_mm256_loadu_ps(pa.add(i+16)),_mm256_loadu_ps(pb.add(i+16)));s2=_mm256_fmadd_ps(d2,d2,s2);
  let d3=_mm256_sub_ps(_mm256_loadu_ps(pa.add(i+24)),_mm256_loadu_ps(pb.add(i+24)));s3=_mm256_fmadd_ps(d3,d3,s3);
  i+=32;}
 let s=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
 let mut t=[0f32;8];_mm256_storeu_ps(t.as_mut_ptr(),s);let mut r:f32=t.iter().sum();
 while i<n{let d=*pa.add(i)-*pb.add(i);r+=d*d;i+=1;}r}
